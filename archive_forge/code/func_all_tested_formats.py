import warnings
import pytest
import numpy as np
from ase import Atoms
from ase.io import write, read, iread
from ase.io.formats import all_formats, ioformats
from ase.calculators.singlepoint import SinglePointCalculator
def all_tested_formats():
    skip = []
    skip += ['dftb', 'eon', 'lammps-data']
    skip += ['v-sim', 'mustem', 'prismatic']
    skip += ['dmol-arc', 'dmol-car', 'dmol-incoor']
    skip += ['gif', 'mp4']
    skip += ['postgresql', 'trj', 'vti', 'vtu', 'mysql']
    if not matplotlib:
        skip += ['eps', 'png']
    if not netCDF4:
        skip += ['netcdftrajectory']
    return sorted(set(all_formats) - set(skip))