from __future__ import annotations
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import polar
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, fast_norm
from pymatgen.core.interface import Interface, label_termination
from pymatgen.core.surface import SlabGenerator
def from_2d_to_3d(mat: np.ndarray) -> np.ndarray:
    """Converts a 2D matrix to a 3D matrix."""
    new_mat = np.diag([1.0, 1.0, 1.0])
    new_mat[:2, :2] = mat
    return new_mat