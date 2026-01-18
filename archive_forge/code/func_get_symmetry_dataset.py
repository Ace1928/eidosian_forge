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
def get_symmetry_dataset(self):
    """Returns the symmetry dataset as a dict.

        Returns:
            dict: With the following properties:
                number: International space group number
                international: International symbol
                hall: Hall symbol
                transformation_matrix: Transformation matrix from lattice of
                input cell to Bravais lattice L^bravais = L^original * Tmat
                origin shift: Origin shift in the setting of "Bravais lattice"
                rotations, translations: Rotation matrices and translation
                vectors. Space group operations are obtained by
                [(r,t) for r, t in zip(rotations, translations)]
                wyckoffs: Wyckoff letters
        """
    return self._space_group_data