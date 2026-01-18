import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def set_fraction_traceless(self, fracTraceless):
    """set what fraction of the traceless part of the force
        on eta is kept.

        By setting this to zero, the volume may change but the shape may not.
        """
    self.frac_traceless = fracTraceless