import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def read_atoms_from_file(fname, fmt):
    assert fname.startswith('"') or fname.startswith("'")
    assert fname[0] == fname[-1]
    fname = fname[1:-1]
    if directory is not None:
        fname = os.path.join(directory, fname)
    if fmt == 'xsf' and 'xsfcoordinatesanimstep' in kwargs:
        anim_step = kwargs.pop('xsfcoordinatesanimstep')
        theslice = slice(anim_step, anim_step + 1, 1)
    else:
        theslice = slice(None, None, 1)
    images = read(fname, theslice, fmt)
    if len(images) != 1:
        raise OctopusParseError("Expected only one image.  Don't know what to do with %d images." % len(images))
    return images[0]