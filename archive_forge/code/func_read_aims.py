import time
import warnings
from ase.units import Ang, fs
from ase.utils import reader, writer
@reader
def read_aims(fd, apply_constraints=True):
    """Import FHI-aims geometry type files.

    Reads unitcell, atom positions and constraints from
    a geometry.in file.

    If geometric constraint (symmetry parameters) are in the file
    include that information in atoms.info["symmetry_block"]
    """
    lines = fd.readlines()
    return parse_geometry_lines(lines, apply_constraints=apply_constraints)