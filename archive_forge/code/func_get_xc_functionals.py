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
def get_xc_functionals(xc_functionals: list | str | None=None) -> list:
    """
        Get XC functionals. If simplified names are provided in kwargs, they
        will be expanded into their corresponding X and C names.
        """
    names = xc_functionals or SETTINGS.get('PMG_DEFAULT_CP2K_FUNCTIONAL')
    if not names:
        raise ValueError('No XC functional provided. Specify kwarg xc_functional or configure PMG_DEFAULT_FUNCTIONAL in your .pmgrc.yaml file')
    if isinstance(names, str):
        names = [names]
    names = [n.upper() for n in names]
    cp2k_names = []
    for name in names:
        if name in ['LDA', 'LSDA']:
            cp2k_names.append('PADE')
        elif name == 'SCAN':
            cp2k_names.extend(['MGGA_X_SCAN', 'MGGA_C_SCAN'])
        elif name == 'SCANL':
            cp2k_names.extend(['MGGA_X_SCANL', 'MGGA_C_SCANL'])
        elif name == 'R2SCAN':
            cp2k_names.extend(['MGGA_X_R2SCAN', 'MGGA_C_R2SCAN'])
        elif name == 'R2SCANL':
            cp2k_names.extend(['MGGA_X_R2SCANL', 'MGGA_C_R2SCANL'])
        else:
            cp2k_names.append(name)
    return cp2k_names