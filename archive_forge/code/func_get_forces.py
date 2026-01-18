import argparse
import traceback
from math import pi
from time import time
import numpy as np
import ase.db
import ase.optimize
from ase.calculators.emt import EMT
from ase.io import Trajectory
def get_forces(self):
    t1 = time()
    f = self.atoms.get_forces()
    if self.eggbox:
        s = np.dot(self.atoms.positions, self.x)
        f += np.dot(np.sin(2 * pi * s), self.x.T) * (2 * pi * self.eggbox / 6)
    t2 = time()
    self.texcl += t2 - t1
    if not self.ready:
        self.nsteps += 1
    self.ready = True
    return f