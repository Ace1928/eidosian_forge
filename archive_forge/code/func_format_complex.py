import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def format_complex(self, val: Union[sympy.Basic, int, float, 'cirq.TParamValComplex']) -> str:
    if isinstance(val, sympy.Basic):
        return str(val)
    c = complex(val)
    joiner = '+'
    abs_imag = c.imag
    if abs_imag < 0:
        joiner = '-'
        abs_imag *= -1
    imag_str = '' if abs_imag == 1 else self.format_real(abs_imag)
    return f'{self.format_real(c.real)}{joiner}{imag_str}i'