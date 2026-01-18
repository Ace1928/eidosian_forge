import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def read_geom(filename, index=':', units=units_CODATA2002):
    """
    Wrapper function for the more generic read() functionality.

    Note that this is function is intended to maintain backwards-compatibility
    only. Keyword arguments will be passed to read_castep_geom().
    """
    from ase.io import read
    return read(filename, index=index, format='castep-geom', units=units)