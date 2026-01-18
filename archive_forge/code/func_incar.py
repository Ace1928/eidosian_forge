from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
@property
def incar(self) -> Incar:
    """Get the INCAR."""
    if self.structure is None:
        raise RuntimeError('No structure is associated with the input set!')
    prev_incar: dict[str, Any] = {}
    if self.inherit_incar is True and self.prev_incar:
        prev_incar = self.prev_incar
    elif isinstance(self.inherit_incar, (list, tuple)) and self.prev_incar:
        prev_incar = {k: self.prev_incar[k] for k in self.inherit_incar if k in self.prev_incar}
    incar_updates = self.incar_updates
    settings = dict(self._config_dict['INCAR'])
    auto_updates = {}
    _apply_incar_updates(settings, incar_updates)
    _apply_incar_updates(settings, self.user_incar_settings)
    structure = self.structure
    comp = structure.composition
    elements = sorted((el for el in comp.elements if comp[el] > 0), key=lambda e: e.X)
    most_electro_neg = elements[-1].symbol
    poscar = Poscar(structure)
    hubbard_u = settings.get('LDAU', False)
    incar = Incar()
    for k, v in settings.items():
        if k == 'MAGMOM':
            mag = []
            for site in structure:
                if hasattr(site, 'magmom'):
                    mag.append(site.magmom)
                elif getattr(site.specie, 'spin', None) is not None:
                    mag.append(site.specie.spin)
                elif str(site.specie) in v:
                    if site.specie.symbol == 'Co' and v[str(site.specie)] <= 1.0:
                        warnings.warn('Co without an oxidation state is initialized as low spin by default in Pymatgen. If this default behavior is not desired, please set the spin on the magmom on the site directly to ensure correct initialization.')
                    mag.append(v.get(str(site.specie)))
                else:
                    if site.specie.symbol == 'Co':
                        warnings.warn('Co without an oxidation state is initialized as low spin by default in Pymatgen. If this default behavior is not desired, please set the spin on the magmom on the site directly to ensure correct initialization.')
                    mag.append(v.get(site.specie.symbol, 0.6))
            incar[k] = mag
        elif k in ('LDAUU', 'LDAUJ', 'LDAUL'):
            if hubbard_u:
                if hasattr(structure[0], k.lower()):
                    m = {site.specie.symbol: getattr(site, k.lower()) for site in structure}
                    incar[k] = [m[sym] for sym in poscar.site_symbols]
                elif most_electro_neg in v and isinstance(v[most_electro_neg], dict):
                    incar[k] = [v[most_electro_neg].get(sym, 0) for sym in poscar.site_symbols]
                else:
                    incar[k] = [v.get(sym, 0) if isinstance(v.get(sym, 0), (float, int)) else 0 for sym in poscar.site_symbols]
        elif k.startswith('EDIFF') and k != 'EDIFFG':
            if 'EDIFF' not in settings and k == 'EDIFF_PER_ATOM':
                incar['EDIFF'] = float(v) * len(structure)
            else:
                incar['EDIFF'] = float(settings['EDIFF'])
        elif k == 'KSPACING' and v == 'auto':
            bandgap = 0 if self.bandgap is None else self.bandgap
            incar[k] = auto_kspacing(bandgap, self.bandgap_tol)
        else:
            incar[k] = v
    has_u = hubbard_u and sum(incar['LDAUU']) > 0
    if not has_u:
        for key in list(incar):
            if key.startswith('LDAU'):
                del incar[key]
    if 'LMAXMIX' not in settings:
        if any((el.Z > 56 for el in structure.composition)):
            incar['LMAXMIX'] = 6
        elif any((el.Z > 20 for el in structure.composition)):
            incar['LMAXMIX'] = 4
    if not incar.get('LASPH', False) and (incar.get('METAGGA') or incar.get('LHFCALC', False) or incar.get('LDAU', False) or incar.get('LUSE_VDW', False)):
        warnings.warn('LASPH = True should be set for +U, meta-GGAs, hybrids, and vdW-DFT', BadInputSetWarning)
    skip = list(self.user_incar_settings) + list(incar_updates)
    skip += ['MAGMOM', 'NUPDOWN', 'LDAUU', 'LDAUL', 'LDAUJ']
    _apply_incar_updates(incar, prev_incar, skip=skip)
    if self.constrain_total_magmom:
        nupdown = sum((mag if abs(mag) > 0.6 else 0 for mag in incar['MAGMOM']))
        if abs(nupdown - round(nupdown)) > 1e-05:
            warnings.warn('constrain_total_magmom was set to True, but the sum of MAGMOM values is not an integer. NUPDOWN is meant to set the spin multiplet and should typically be an integer. You are likely better off changing the values of MAGMOM or simply setting NUPDOWN directly in your INCAR settings.', UserWarning, stacklevel=1)
        auto_updates['NUPDOWN'] = nupdown
    if self.use_structure_charge:
        auto_updates['NELECT'] = self.nelect
    if incar.get('LHFCALC', False) is True and incar.get('ALGO', 'Normal') not in ['Normal', 'All', 'Damped']:
        warnings.warn('Hybrid functionals only support Algo = All, Damped, or Normal.', BadInputSetWarning)
    if self.auto_ismear:
        if self.bandgap is None:
            auto_updates.update(ISMEAR=2, SIGMA=0.2)
        elif self.bandgap <= self.bandgap_tol:
            auto_updates.update(ISMEAR=2, SIGMA=0.2)
        else:
            auto_updates.update(ISMEAR=-5, SIGMA=0.05)
    kpoints = self.kpoints
    if kpoints is not None:
        incar.pop('KSPACING', None)
    elif 'KSPACING' in incar and 'KSPACING' not in self.user_incar_settings and ('KSPACING' in prev_incar):
        incar['KSPACING'] = prev_incar['KSPACING']
    _apply_incar_updates(incar, auto_updates, skip=list(self.user_incar_settings))
    _remove_unused_incar_params(incar, skip=list(self.user_incar_settings))
    if kpoints is not None and np.prod(kpoints.kpts) < 4 and (incar.get('ISMEAR', 0) == -5):
        incar['ISMEAR'] = 0
    if self.user_incar_settings.get('KSPACING', 0) > 0.5 and incar.get('ISMEAR', 0) == -5:
        warnings.warn('Large KSPACING value detected with ISMEAR = -5. Ensure that VASP generates an adequate number of KPOINTS, lower KSPACING, or set ISMEAR = 0', BadInputSetWarning)
    ismear = incar.get('ISMEAR', 1)
    sigma = incar.get('SIGMA', 0.2)
    if all((elem.is_metal for elem in structure.composition)) and incar.get('NSW', 0) > 0 and (ismear < 0 or (ismear == 0 and sigma > 0.05)):
        ismear_docs = 'https://www.vasp.at/wiki/index.php/ISMEAR'
        msg = ''
        if ismear < 0:
            msg = f'Relaxation of likely metal with ISMEAR < 0 ({ismear}).'
        elif ismear == 0 and sigma > 0.05:
            msg = f'ISMEAR = 0 with a small SIGMA ({sigma}) detected.'
        warnings.warn(f'{msg} See VASP recommendations on ISMEAR for metals ({ismear_docs}).', BadInputSetWarning, stacklevel=1)
    return incar