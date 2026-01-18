import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import jsonio
from ase.io.ulm import open as ulmopen
from ase.parallel import paropen, world, barrier
from ase.calculators.singlepoint import (SinglePointCalculator,
def read_extra_data(self, name, n=0):
    """Read extra data stored alongside the atoms.

        Currently only used to read data stored by an NPT dynamics object.
        The data is not associated with individual atoms.
        """
    if self.state != 'read':
        raise IOError('Cannot read extra data in %s mode' % (self.state,))
    if n < 0:
        n += self.nframes
    if n < 0 or n >= self.nframes:
        raise IndexError('Trajectory index %d out of range [0, %d[' % (n, self.nframes))
    framedir = os.path.join(self.filename, 'F' + str(n))
    framezero = os.path.join(self.filename, 'F0')
    return self._read_data(framezero, framedir, name, self.atom_id)