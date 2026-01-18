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
def read_bundletrajectory(filename, index=-1):
    """Reads one or more atoms objects from a BundleTrajectory.

    Arguments:

    filename: str
        The name of the bundle (really a directory!)
    index: int
        An integer specifying which frame to read, or an index object
        for reading multiple frames.  Default: -1 (reads the last
        frame).
    """
    traj = BundleTrajectory(filename, mode='r')
    for i in range(*index.indices(len(traj))):
        yield traj[i]