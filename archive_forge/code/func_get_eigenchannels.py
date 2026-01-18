import numpy as np
from numpy import linalg
from ase.transport.selfenergy import LeadSelfEnergy, BoxProbe
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import subdiagonalize, cutcoupling, dagger,\
from ase.units import kB
def get_eigenchannels(self, n=None):
    """Get ``n`` first eigenchannels."""
    self.initialize()
    self.update()
    if n is None:
        n = self.input_parameters['eigenchannels']
    return self.eigenchannels_ne[:n]