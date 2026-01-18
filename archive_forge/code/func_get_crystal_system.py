from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
def get_crystal_system(self) -> CrystalSystem:
    """Get the crystal system for the structure, e.g., (triclinic, orthorhombic,
        cubic, etc.).

        Raises:
            ValueError: on invalid space group numbers < 1 or > 230.

        Returns:
            str: Crystal system for structure
        """
    n = self._space_group_data['number']
    if not (n == int(n) and 0 < n < 231):
        raise ValueError(f'Received invalid space group {n}')
    if 0 < n < 3:
        return 'triclinic'
    if n < 16:
        return 'monoclinic'
    if n < 75:
        return 'orthorhombic'
    if n < 143:
        return 'tetragonal'
    if n < 168:
        return 'trigonal'
    if n < 195:
        return 'hexagonal'
    return 'cubic'