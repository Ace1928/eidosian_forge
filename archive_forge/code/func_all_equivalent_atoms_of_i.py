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
def all_equivalent_atoms_of_i(idx, eq_sets, ops):
    """WORKS INPLACE on operations."""
    visited = {idx}
    tmp_eq_sets = {j: eq_sets[j] - visited for j in eq_sets[idx]}
    while tmp_eq_sets:
        new_tmp_eq_sets = {}
        for j in tmp_eq_sets:
            if j in visited:
                continue
            visited.add(j)
            for k in tmp_eq_sets[j]:
                new_tmp_eq_sets[k] = eq_sets[k] - visited
                if idx not in ops[k]:
                    ops[k][idx] = np.dot(ops[j][idx], ops[k][j]) if k != idx else unit_mat
                ops[idx][k] = ops[k][idx].T
        tmp_eq_sets = new_tmp_eq_sets
    return (visited, ops)