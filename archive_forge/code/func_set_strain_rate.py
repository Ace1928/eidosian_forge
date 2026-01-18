import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def set_strain_rate(self, rate):
    """Set the strain rate.  Must be an upper triangular 3x3 matrix.

        If you set a strain rate along a direction that is "masked out"
        (see ``set_mask``), the strain rate along that direction will be
        maintained constantly.
        """
    if not (rate.shape == (3, 3) and self._isuppertriangular(rate)):
        raise ValueError('Strain rate must be an upper triangular matrix.')
    self.eta = rate
    if self.initialized:
        self._initialize_eta_h()