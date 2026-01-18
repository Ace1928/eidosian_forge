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
@staticmethod
def parse_transformation_string(transformation_string: str='a,b,c;0,0,0') -> tuple[list[list[float]] | np.ndarray, list[float]]:
    """
        Args:
            transformation_string (str, optional): Defaults to "a,b,c;0,0,0".

        Raises:
            ValueError: When transformation string fails to parse.

        Returns:
            tuple[list[list[float]] | np.ndarray, list[float]]: transformation matrix & vector
        """
    try:
        a, b, c = np.eye(3)
        b_change, o_shift = transformation_string.split(';')
        basis_change = b_change.split(',')
        origin_shift = o_shift.split(',')
        basis_change = [re.sub('(?<=\\w|\\))(?=\\() | (?<=\\))(?=\\w) | (?<=(\\d|a|b|c))(?=([abc]))', '*', string, flags=re.VERBOSE) for string in basis_change]
        allowed_chars = '0123456789+-*/.abc()'
        basis_change = [''.join([c for c in string if c in allowed_chars]) for string in basis_change]
        basis_change = [parse_expr(string).subs({'a': Matrix(a), 'b': Matrix(b), 'c': Matrix(c)}) for string in basis_change]
        P = np.array(basis_change, dtype=float).T[0]
        p = [float(Fraction(x)) for x in origin_shift]
        return (P, p)
    except Exception as exc:
        raise ValueError(f'Failed to parse transformation string: {exc}')