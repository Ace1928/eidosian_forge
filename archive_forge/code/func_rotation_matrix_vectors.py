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
def rotation_matrix_vectors(v1, v2):
    """Returns the rotation matrix that rotates v1 onto v2 using
        Rodrigues' rotation formula.

        See more: https://math.stackexchange.com/a/476311

        Args:
            v1: initial vector
            v2: target vector

        Returns:
            3x3 rotation matrix
        """
    if np.allclose(v1, v2):
        return np.eye(3)
    if np.allclose(v1, -v2):
        return np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    v = np.cross(v1, v2)
    norm = np.linalg.norm(v)
    c = np.vdot(v1, v2)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + np.dot(vx, vx) * ((1.0 - c) / (norm * norm))