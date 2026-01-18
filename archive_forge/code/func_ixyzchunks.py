from itertools import islice
import re
import warnings
from io import StringIO, UnsupportedOperation
import json
import numpy as np
import numbers
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from ase.utils import reader
def ixyzchunks(fd):
    """Yield unprocessed chunks (header, lines) for each xyz image."""
    while True:
        line = next(fd).strip()
        try:
            natoms = int(line)
        except ValueError:
            raise XYZError('Expected integer, found "{0}"'.format(line))
        try:
            lines = [next(fd) for _ in range(1 + natoms)]
        except StopIteration:
            raise XYZError('Incomplete XYZ chunk')
        yield XYZChunk(lines, natoms)