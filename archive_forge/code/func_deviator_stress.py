from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.tensors import SquareTensor
@property
def deviator_stress(self):
    """Returns the deviatoric component of the stress."""
    if not self.is_symmetric:
        raise ValueError('The stress tensor is not symmetric, so deviator stress will not be either')
    return self - self.mean_stress * np.eye(3)