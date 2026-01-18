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
def best_case(self, matrix, m_list, indices_left):
    """
        Computes a best case given a matrix and manipulation list.

        Args:
            matrix: the current matrix (with some permutations already
                performed)
            m_list: [(multiplication fraction, number_of_indices, indices,
                species)] describing the manipulation
            indices: Set of indices which haven't had a permutation
                performed on them.
        """
    m_indices = []
    fraction_list = []
    for m in m_list:
        m_indices.extend(m[2])
        fraction_list.extend([m[0]] * m[1])
    indices = list(indices_left.intersection(m_indices))
    interaction_matrix = matrix[indices, :][:, indices]
    fractions = np.zeros(len(interaction_matrix)) + 1
    fractions[:len(fraction_list)] = fraction_list
    fractions = np.sort(fractions)
    sums = 2 * np.sum(matrix[indices], axis=1)
    sums = np.sort(sums)
    step1 = np.sort(interaction_matrix) * (1 - fractions)
    step2 = np.sort(np.sum(step1, axis=1))
    step3 = step2 * (1 - fractions)
    interaction_correction = np.sum(step3)
    if self._algo == self.ALGO_TIME_LIMIT:
        elapsed_time = datetime.utcnow() - self._start_time
        speedup_parameter = elapsed_time.total_seconds() / 1800
        avg_int = np.sum(interaction_matrix, axis=None)
        avg_frac = np.average(np.outer(1 - fractions, 1 - fractions))
        average_correction = avg_int * avg_frac
        interaction_correction = average_correction * speedup_parameter + interaction_correction * (1 - speedup_parameter)
    return np.sum(matrix) + np.inner(sums[::-1], fractions - 1) + interaction_correction