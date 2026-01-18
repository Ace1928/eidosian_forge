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
@staticmethod
def is_empty_bundle(filename):
    """Check if a filename is an empty bundle.

        Assumes that it is a bundle."""
    if not os.listdir(filename):
        return True
    with open(os.path.join(filename, 'frames'), 'rb') as fd:
        nframes = int(fd.read())
    barrier()
    return nframes == 0