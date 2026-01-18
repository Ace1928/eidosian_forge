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
def get_point_group_symbol(self) -> str:
    """Get the point group associated with the structure.

        Returns:
            Pointgroup: Point group for structure.
        """
    rotations = self._space_group_data['rotations']
    if len(rotations) == 0:
        return '1'
    return spglib.get_pointgroup(rotations)[0].strip()