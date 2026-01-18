from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
def minimize_matrix(self):
    """
        This method finds and returns the permutations that produce the lowest
        Ewald sum calls recursive function to iterate through permutations.
        """
    if self._algo in (EwaldMinimizer.ALGO_FAST, EwaldMinimizer.ALGO_BEST_FIRST):
        return self._recurse(self._matrix, self._m_list, set(range(len(self._matrix))))
    return None