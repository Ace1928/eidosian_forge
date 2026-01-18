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
def is_bundle(filename, allowempty=False):
    """Check if a filename exists and is a BundleTrajectory.

        If allowempty=True, an empty folder is regarded as an
        empty BundleTrajectory."""
    filename = Path(filename)
    if not filename.is_dir():
        return False
    if allowempty and (not os.listdir(filename)):
        return True
    metaname = filename / 'metadata.json'
    if metaname.is_file():
        mdata = jsonio.decode(metaname.read_text())
    else:
        return False
    try:
        return mdata['format'] == 'BundleTrajectory'
    except KeyError:
        return False