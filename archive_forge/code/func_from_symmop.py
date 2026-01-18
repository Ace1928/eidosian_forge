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
def from_symmop(cls, symmop: SymmOp, time_reversal) -> Self:
    """Initialize a MagSymmOp from a SymmOp and time reversal operator.

        Args:
            symmop (SymmOp): SymmOp
            time_reversal (int): Time reversal operator, +1 or -1.

        Returns:
            MagSymmOp object
        """
    return cls(symmop.affine_matrix, time_reversal, symmop.tol)