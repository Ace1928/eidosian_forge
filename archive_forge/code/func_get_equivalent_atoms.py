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
def get_equivalent_atoms(self):
    """Returns sets of equivalent atoms with symmetry operations.

        Returns:
            dict: with two possible keys:
                eq_sets: A dictionary of indices mapping to sets of indices, each key maps to
                    indices of all equivalent atoms. The keys are guaranteed to be not equivalent.
                sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry
                    operation that maps atom i unto j.
        """
    eq = self._get_eq_sets()
    return self._combine_eq_sets(eq['eq_sets'], eq['sym_ops'])