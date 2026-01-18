from __future__ import annotations
import re
import string
import typing
import warnings
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.due import Doi, due
from pymatgen.util.string import transformation_to_string
@classmethod
def from_rotation_and_translation(cls, rotation_matrix: ArrayLike=((1, 0, 0), (0, 1, 0), (0, 0, 1)), translation_vec: ArrayLike=(0, 0, 0), tol: float=0.1) -> Self:
    """Creates a symmetry operation from a rotation matrix and a translation
        vector.

        Args:
            rotation_matrix (3x3 array): Rotation matrix.
            translation_vec (3x1 array): Translation vector.
            tol (float): Tolerance to determine if rotation matrix is valid.

        Returns:
            SymmOp object
        """
    rotation_matrix = np.array(rotation_matrix)
    translation_vec = np.array(translation_vec)
    if rotation_matrix.shape != (3, 3):
        raise ValueError('Rotation Matrix must be a 3x3 numpy array.')
    if translation_vec.shape != (3,):
        raise ValueError('Translation vector must be a rank 1 numpy array with 3 elements.')
    affine_matrix = np.eye(4)
    affine_matrix[0:3][:, 0:3] = rotation_matrix
    affine_matrix[0:3][:, 3] = translation_vec
    return cls(affine_matrix, tol)