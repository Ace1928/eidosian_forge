from __future__ import annotations
import itertools
import os
import warnings
import numpy as np
from ruamel.yaml import YAML
from pymatgen.core import SETTINGS
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Element, Molecule, Structure
from pymatgen.io.cp2k.inputs import (
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff, get_unique_site_indices
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
@staticmethod
def get_basis_and_potential(structure, basis_and_potential):
    """
        Get a dictionary of basis and potential info for constructing the input file.

        data in basis_and_potential argument can be specified in several ways:

            Strategy 1: Element-specific info (takes precedence)

                1. Provide a basis and potential object:

                    el: {'basis': obj, 'potential': obj}

                2. Provide a hash of the object that matches the keys in the pmg configured cp2k data files.

                    el: {'basis': hash, 'potential': hash}

                3. Provide the name of the basis and potential AND the basis_filenames and potential_filename
                keywords specifying where to find these objects

                    el: {
                        'basis': name, 'potential': name, 'basis_filenames': [filenames],
                        'potential_filename': filename
                    }

            Strategy 2: global descriptors

                In this case, any elements not present in the argument will be dealt with by searching the pmg
                configured cp2k data files to find a objects matching your requirements.

                    - functional: Find potential and basis that have been optimized for a specific functional like PBE.
                        Can be None if you do not require them to match.
                    - basis_type: type of basis to search for (e.g. DZVP-MOLOPT).
                    - aux_basis_type: type of basis to search for (e.g. pFIT). Some elements do not have all aux types
                        available. Use aux_basis_type that is universal to avoid issues, or avoid using this strategy.
                    - potential_type: "Pseudopotential" or "All Electron"

                ***BE WARNED*** CP2K data objects can have the same name, this will sort those and choose the first one
                that matches.

        Will raise an error if no basis/potential info can be found according to the input.
        """
    data = {'basis_filenames': []}
    functional = basis_and_potential.get('functional', SETTINGS.get('PMG_DEFAULT_CP2K_FUNCTIONAL'))
    basis_type = basis_and_potential.get('basis_type', SETTINGS.get('PMG_DEFAULT_CP2K_BASIS_TYPE'))
    potential_type = basis_and_potential.get('potential_type', SETTINGS.get('PMG_DEFAULT_POTENTIAL_TYPE', 'Pseudopotential'))
    aux_basis_type = basis_and_potential.get('aux_basis_type', SETTINGS.get('PMG_DEFAULT_CP2K_AUX_BASIS_TYPE'))
    for el in structure.symbol_set:
        possible_basis_sets = []
        possible_potentials = []
        basis, aux_basis, potential = (None, None, None)
        desired_basis, desired_aux_basis, desired_potential = (None, None, None)
        have_element_file = os.path.isfile(os.path.join(SETTINGS.get('PMG_CP2K_DATA_DIR', '.'), el))
        if have_element_file:
            with open(os.path.join(SETTINGS.get('PMG_CP2K_DATA_DIR', '.'), el), encoding='utf-8') as file:
                yaml = YAML(typ='unsafe', pure=True)
                DATA = yaml.load(file)
                if not DATA.get('basis_sets'):
                    raise ValueError(f'No standard basis sets available in data directory for {el}')
                if not DATA.get('potentials'):
                    raise ValueError(f'No standard potentials available in data directory for {el}')
        if el in basis_and_potential:
            _basis = basis_and_potential[el].get('basis')
            if isinstance(_basis, GaussianTypeOrbitalBasisSet):
                possible_basis_sets.append(_basis)
            elif have_element_file:
                if _basis in DATA['basis_sets']:
                    possible_basis_sets.append(GaussianTypeOrbitalBasisSet.from_dict(DATA['basis_sets'][_basis]))
                elif _basis:
                    desired_basis = GaussianTypeOrbitalBasisSet(name=_basis)
            _aux_basis = basis_and_potential[el].get('aux_basis')
            if isinstance(_aux_basis, GaussianTypeOrbitalBasisSet):
                aux_basis = _aux_basis
            elif have_element_file:
                if _aux_basis in DATA['basis_sets']:
                    aux_basis = GaussianTypeOrbitalBasisSet.from_dict(DATA['basis_sets'][_aux_basis])
                elif _aux_basis:
                    desired_aux_basis = GaussianTypeOrbitalBasisSet(name=_aux_basis)
            _potential = basis_and_potential[el].get('potential')
            if isinstance(_potential, GthPotential):
                possible_potentials.append(_potential)
            elif have_element_file:
                if _potential in DATA['potentials']:
                    possible_potentials.append(GthPotential.from_dict(DATA['potentials'][_potential]))
                elif _potential:
                    desired_potential = GthPotential(name=_potential)
            if basis_and_potential[el].get('basis_filename'):
                data['basis_filenames'].append(basis_and_potential[el].get('basis_filename'))
            pfn1 = basis_and_potential[el].get('potential_filename')
            pfn2 = data.get('potential_filename')
            if pfn1 and pfn2 and (pfn1 != pfn2):
                raise ValueError('Provided potentials have more than one corresponding file.CP2K does not support multiple potential filenames')
            data['potential_filename'] = basis_and_potential[el].get('potential_filename')
        else:
            if basis_type and have_element_file:
                desired_basis = GaussianTypeOrbitalBasisSet(element=Element(el), potential=potential_type, info=BasisInfo.from_str(f'{basis_type}-{functional}'))
                desired_potential = GthPotential(element=Element(el), potential=potential_type, info=PotentialInfo(xc=functional))
            if aux_basis_type and have_element_file:
                desired_aux_basis = GaussianTypeOrbitalBasisSet(info=BasisInfo.from_str(aux_basis_type))
        if desired_basis:
            for _possible_basis in DATA.get('basis_sets').values():
                possible_basis = GaussianTypeOrbitalBasisSet.from_dict(_possible_basis)
                if desired_basis.softmatch(possible_basis):
                    possible_basis_sets.append(possible_basis)
        if desired_aux_basis:
            for _possible_basis in DATA.get('basis_sets').values():
                possible_basis = GaussianTypeOrbitalBasisSet.from_dict(_possible_basis)
                if desired_aux_basis.softmatch(possible_basis):
                    aux_basis = possible_basis
                    data['basis_filenames'].append(aux_basis.filename)
                    break
        if desired_potential:
            for _possible_potential in DATA.get('potentials').values():
                possible_potential = GthPotential.from_dict(_possible_potential)
                if desired_potential.softmatch(possible_potential):
                    possible_potentials.append(possible_potential)
        possible_basis_sets = sorted(filter(lambda x: x.info.electrons, possible_basis_sets), key=lambda x: x.info.electrons, reverse=True)
        possible_potentials = sorted(filter(lambda x: x.info.electrons, possible_potentials), key=lambda x: x.info.electrons, reverse=True)

        def match_elecs(x):
            for p in possible_potentials:
                if x.info.electrons == p.info.electrons:
                    return p
            return None
        for b in possible_basis_sets:
            fb = match_elecs(b)
            if fb is not None:
                basis = b
                potential = fb
                break
        if basis is None:
            if not basis_and_potential.get(el, {}).get('basis'):
                raise ValueError(f'No explicit basis found for {el} and matching has failed.')
            warnings.warn(f'Unable to validate basis for {el}. Exact name provided will be put in input file.')
            basis = basis_and_potential[el].get('basis')
        if aux_basis is None and basis_and_potential.get(el, {}).get('aux_basis'):
            warnings.warn(f'Unable to validate auxiliary basis for {el}. Exact name provided will be put in input file.')
            aux_basis = basis_and_potential[el].get('aux_basis')
        if potential is None:
            if basis_and_potential.get(el, {}).get('potential'):
                warnings.warn(f'Unable to validate potential for {el}. Exact name provided will be put in input file.')
                potential = basis_and_potential.get(el, {}).get('potential')
            else:
                raise ValueError('No explicit potential found and matching has failed.')
        if hasattr(basis, 'filename'):
            data['basis_filenames'].append(basis.filename)
        pfn1 = data.get('potential_filename')
        pfn2 = potential.filename
        if pfn1 and pfn2 and (pfn1 != pfn2):
            raise ValueError('Provided potentials have more than one corresponding file.CP2K does not support multiple potential filenames')
        data['potential_filename'] = pfn2
        data[el] = {'basis': basis, 'aux_basis': aux_basis, 'potential': potential}
    return data