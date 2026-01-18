import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def set_ax1_range(self, ylim):
    self._ax1_ylim = ylim
    self.ax1.set_ylim(ylim)