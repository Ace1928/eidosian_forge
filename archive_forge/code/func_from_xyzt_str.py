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
@classmethod
def from_xyzt_str(cls, xyzt_str: str) -> Self:
    """
        Args:
            xyzt_str (str): of the form 'x, y, z, +1', '-x, -y, z, -1',
                '-2y+1/2, 3x+1/2, z-y+1/2, +1', etc.

        Returns:
            MagSymmOp object
        """
    symm_op = SymmOp.from_xyz_str(xyzt_str.rsplit(',', 1)[0])
    try:
        time_reversal = int(xyzt_str.rsplit(',', 1)[1])
    except Exception:
        raise RuntimeError('Time reversal operator could not be parsed.')
    return cls.from_symmop(symm_op, time_reversal)