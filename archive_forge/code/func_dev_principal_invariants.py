from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.tensors import SquareTensor
@property
def dev_principal_invariants(self):
    """
        Returns the principal invariants of the deviatoric stress tensor,
        which is calculated by finding the coefficients of the characteristic
        polynomial of the stress tensor minus the identity times the mean
        stress.
        """
    return self.deviator_stress.principal_invariants * np.array([1, -1, 1])