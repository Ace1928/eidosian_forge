from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.tensors import SquareTensor
def piola_kirchoff_1(self, def_grad):
    """
        Calculates the first Piola-Kirchoff stress.

        Args:
            def_grad (3x3 array-like): deformation gradient tensor
        """
    if not self.is_symmetric:
        raise ValueError('The stress tensor is not symmetric, PK stress is based on a symmetric stress tensor.')
    def_grad = SquareTensor(def_grad)
    return def_grad.det * np.dot(self, def_grad.inv.trans)