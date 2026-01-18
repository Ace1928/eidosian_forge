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
def get_symmetrized_structure(self):
    """Get a symmetrized structure. A symmetrized structure is one where the sites
        have been grouped into symmetrically equivalent groups.

        Returns:
            pymatgen.symmetry.structure.SymmetrizedStructure object.
        """
    sym_dataset = self.get_symmetry_dataset()
    spg_ops = SpacegroupOperations(self.get_space_group_symbol(), self.get_space_group_number(), self.get_symmetry_operations())
    return SymmetrizedStructure(self._structure, spg_ops, sym_dataset['equivalent_atoms'], sym_dataset['wyckoffs'])