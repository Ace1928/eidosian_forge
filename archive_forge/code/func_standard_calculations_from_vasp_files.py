from __future__ import annotations
import itertools
import os
import re
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any
import numpy as np
import spglib
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
@classmethod
def standard_calculations_from_vasp_files(cls, POSCAR_input: str='POSCAR', INCAR_input: str='INCAR', POTCAR_input: str | None=None, Vasprun_output: str='vasprun.xml', dict_for_basis: dict | None=None, option: str='standard'):
    """
        Will generate Lobsterin with standard settings.

        Args:
            POSCAR_input (str): path to POSCAR
            INCAR_input (str): path to INCAR
            POTCAR_input (str): path to POTCAR
            dict_for_basis (dict): can be provided: it should look the following:
                dict_for_basis={"Fe":'3p 3d 4s 4f', "C": '2s 2p'} and will overwrite all settings from POTCAR_input

            option (str): 'standard' will start a normal lobster run where COHPs, COOPs, DOS, CHARGE etc. will be
                calculated
                'standard_with_energy_range_from_vasprun' will start a normal lobster run for entire energy range
                of VASP static run. vasprun.xml file needs to be in current directory.
                'standard_from_projection' will start a normal lobster run from a projection
                'standard_with_fatband' will do a fatband calculation, run over all orbitals
                'onlyprojection' will only do a projection
                'onlydos' will only calculate a projected dos
                'onlycohp' will only calculate cohp
                'onlycoop' will only calculate coop
                'onlycohpcoop' will only calculate cohp and coop

        Returns:
            Lobsterin Object with standard settings
        """
    warnings.warn('Always check and test the provided basis functions. The spilling of your Lobster calculation might help')
    if option not in ['standard', 'standard_from_projection', 'standard_with_fatband', 'standard_with_energy_range_from_vasprun', 'onlyprojection', 'onlydos', 'onlycohp', 'onlycoop', 'onlycobi', 'onlycohpcoop', 'onlycohpcoopcobi', 'onlymadelung']:
        raise ValueError('The option is not valid!')
    Lobsterindict: dict[str, Any] = {'basisSet': 'pbeVaspFit2015', 'COHPstartEnergy': -35.0, 'COHPendEnergy': 5.0}
    if option in {'standard', 'standard_with_energy_range_from_vasprun', 'onlycohp', 'onlycoop', 'onlycobi', 'onlycohpcoop', 'onlycohpcoopcobi', 'standard_with_fatband'}:
        Lobsterindict['cohpGenerator'] = 'from 0.1 to 6.0 orbitalwise'
        Lobsterindict['saveProjectionToFile'] = True
    if option == 'standard_from_projection':
        Lobsterindict['cohpGenerator'] = 'from 0.1 to 6.0 orbitalwise'
        Lobsterindict['loadProjectionFromFile'] = True
    if option == 'standard_with_energy_range_from_vasprun':
        Vr = Vasprun(Vasprun_output)
        Lobsterindict['COHPstartEnergy'] = round(min(Vr.complete_dos.energies - Vr.complete_dos.efermi), 4)
        Lobsterindict['COHPendEnergy'] = round(max(Vr.complete_dos.energies - Vr.complete_dos.efermi), 4)
        Lobsterindict['COHPSteps'] = len(Vr.complete_dos.energies)
    if option == 'onlycohp':
        Lobsterindict['skipdos'] = True
        Lobsterindict['skipcoop'] = True
        Lobsterindict['skipPopulationAnalysis'] = True
        Lobsterindict['skipGrossPopulation'] = True
        Lobsterindict['skipcobi'] = True
        Lobsterindict['skipMadelungEnergy'] = True
    if option == 'onlycoop':
        Lobsterindict['skipdos'] = True
        Lobsterindict['skipcohp'] = True
        Lobsterindict['skipPopulationAnalysis'] = True
        Lobsterindict['skipGrossPopulation'] = True
        Lobsterindict['skipcobi'] = True
        Lobsterindict['skipMadelungEnergy'] = True
    if option == 'onlycohpcoop':
        Lobsterindict['skipdos'] = True
        Lobsterindict['skipPopulationAnalysis'] = True
        Lobsterindict['skipGrossPopulation'] = True
        Lobsterindict['skipcobi'] = True
        Lobsterindict['skipMadelungEnergy'] = True
    if option == 'onlycohpcoopcobi':
        Lobsterindict['skipdos'] = True
        Lobsterindict['skipPopulationAnalysis'] = True
        Lobsterindict['skipGrossPopulation'] = True
        Lobsterindict['skipMadelungEnergy'] = True
    if option == 'onlydos':
        Lobsterindict['skipcohp'] = True
        Lobsterindict['skipcoop'] = True
        Lobsterindict['skipPopulationAnalysis'] = True
        Lobsterindict['skipGrossPopulation'] = True
        Lobsterindict['skipcobi'] = True
        Lobsterindict['skipMadelungEnergy'] = True
    if option == 'onlyprojection':
        Lobsterindict['skipdos'] = True
        Lobsterindict['skipcohp'] = True
        Lobsterindict['skipcoop'] = True
        Lobsterindict['skipPopulationAnalysis'] = True
        Lobsterindict['skipGrossPopulation'] = True
        Lobsterindict['saveProjectionToFile'] = True
        Lobsterindict['skipcobi'] = True
        Lobsterindict['skipMadelungEnergy'] = True
    if option == 'onlycobi':
        Lobsterindict['skipdos'] = True
        Lobsterindict['skipcohp'] = True
        Lobsterindict['skipPopulationAnalysis'] = True
        Lobsterindict['skipGrossPopulation'] = True
        Lobsterindict['skipcobi'] = True
        Lobsterindict['skipMadelungEnergy'] = True
    if option == 'onlymadelung':
        Lobsterindict['skipdos'] = True
        Lobsterindict['skipcohp'] = True
        Lobsterindict['skipcoop'] = True
        Lobsterindict['skipPopulationAnalysis'] = True
        Lobsterindict['skipGrossPopulation'] = True
        Lobsterindict['saveProjectionToFile'] = True
        Lobsterindict['skipcobi'] = True
    incar = Incar.from_file(INCAR_input)
    if incar['ISMEAR'] == 0:
        Lobsterindict['gaussianSmearingWidth'] = incar['SIGMA']
    if incar['ISMEAR'] != 0 and option == 'standard_with_fatband':
        raise ValueError('ISMEAR has to be 0 for a fatband calculation with Lobster')
    if dict_for_basis is not None:
        basis = [f'{key} {value}' for key, value in dict_for_basis.items()]
    elif POTCAR_input is not None:
        potcar_names = Lobsterin._get_potcar_symbols(POTCAR_input=POTCAR_input)
        basis = Lobsterin.get_basis(structure=Structure.from_file(POSCAR_input), potcar_symbols=potcar_names)
    else:
        raise ValueError('basis cannot be generated')
    Lobsterindict['basisfunctions'] = basis
    if option == 'standard_with_fatband':
        Lobsterindict['createFatband'] = basis
    return cls(Lobsterindict)