from __future__ import annotations
import re
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from sympy import Matrix
from sympy.parsing.sympy_parser import parse_expr
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.util.string import transformation_to_string
@classmethod
def from_origin_shift(cls, origin_shift: str='0,0,0') -> Self:
    """Construct SpaceGroupTransformation from its origin shift string.

        Args:
            origin_shift (str, optional): Defaults to "0,0,0".

        Returns:
            JonesFaithfulTransformation
        """
    P = np.identity(3)
    p = [float(Fraction(x)) for x in origin_shift.split(',')]
    return cls(P, p)