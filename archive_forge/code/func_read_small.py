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
def read_small(self, framedir):
    """Read small data."""
    with ulmopen(os.path.join(framedir, 'smalldata.ulm'), 'r') as fd:
        return fd.asdict()