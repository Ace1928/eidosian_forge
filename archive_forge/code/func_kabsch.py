from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
@staticmethod
def kabsch(P: np.ndarray, Q: np.ndarray):
    """The Kabsch algorithm is a method for calculating the optimal rotation matrix
        that minimizes the root mean squared deviation (RMSD) between two paired sets of points
        P and Q, centered around the their centroid.

        For more info see:
        - http://wikipedia.org/wiki/Kabsch_algorithm and
        - https://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures

        Args:
            P: Nx3 matrix, where N is the number of points.
            Q: Nx3 matrix, where N is the number of points.

        Returns:
            U: 3x3 rotation matrix
        """
    C = np.dot(P.T, Q)
    V, _S, WT = np.linalg.svd(C)
    det = np.linalg.det(np.dot(V, WT))
    return np.dot(np.dot(V, np.diag([1, 1, det])), WT)