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
def read_split(self, framedir, name):
    """Read data from multiple files.

        Falls back to reading from single file if that is how data is stored.

        Returns the data and an object indicating if the data was really
        read from split files.  The latter object is False if not
        read from split files, but is an array of the segment length if
        split files were used.
        """
    data = []
    if os.path.exists(os.path.join(framedir, name + '.ulm')):
        return (self.read(framedir, name), False)
    for i in range(self.nfrag):
        suf = '_%d' % (i,)
        data.append(self.read(framedir, name + suf))
    seglengths = [len(d) for d in data]
    return (np.concatenate(data), seglengths)