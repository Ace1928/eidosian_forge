import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
@reader
def read_vasp_out(filename='OUTCAR', index=-1):
    """Import OUTCAR type file.

    Reads unitcell, atom positions, energies, and forces from the OUTCAR file
    and attempts to read constraints (if any) from CONTCAR/POSCAR, if present.
    """
    g = iread_vasp_out(filename, index=index)
    if isinstance(index, (slice, str)):
        return list(g)
    else:
        return next(g)