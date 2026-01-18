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
def print_mo_cubes(self, write_cube: bool=False, nlumo: int=-1, nhomo: int=-1) -> None:
    """
        Activate printing of molecular orbitals.

        Args:
            write_cube (bool): whether to write cube file for the MOs instead of out file
            nlumo (int): Controls the number of lumos printed and dumped as a cube (-1=all)
            nhomo (int): Controls the number of homos printed and dumped as a cube (-1=all)
        """
    if not self.check('FORCE_EVAL/DFT/PRINT/MO_CUBES'):
        self['FORCE_EVAL']['DFT']['PRINT'].insert(MOCubes(write_cube=write_cube, nlumo=nlumo, nhomo=nhomo))