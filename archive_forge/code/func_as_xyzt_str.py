from __future__ import annotations
import re
import string
import typing
import warnings
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.due import Doi, due
from pymatgen.util.string import transformation_to_string
def as_xyzt_str(self) -> str:
    """Returns a string of the form 'x, y, z, +1', '-x, -y, z, -1',
        '-y+1/2, x+1/2, z+1/2, +1', etc. Only works for integer rotation matrices.
        """
    xyzt_string = SymmOp.as_xyz_str(self)
    return f'{xyzt_string}, {self.time_reversal:+}'