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
@staticmethod
def from_rotation_and_translation_and_time_reversal(rotation_matrix: ArrayLike=((1, 0, 0), (0, 1, 0), (0, 0, 1)), translation_vec: ArrayLike=(0, 0, 0), time_reversal: int=1, tol: float=0.1) -> MagSymmOp:
    """Creates a symmetry operation from a rotation matrix, translation
        vector and time reversal operator.

        Args:
            rotation_matrix (3x3 array): Rotation matrix.
            translation_vec (3x1 array): Translation vector.
            time_reversal (int): Time reversal operator, +1 or -1.
            tol (float): Tolerance to determine if rotation matrix is valid.

        Returns:
            MagSymmOp object
        """
    symm_op = SymmOp.from_rotation_and_translation(rotation_matrix=rotation_matrix, translation_vec=translation_vec, tol=tol)
    return MagSymmOp.from_symmop(symm_op, time_reversal)