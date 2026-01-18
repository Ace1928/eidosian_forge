from typing import (
import numbers
import sympy
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import (
@property
def exponent_relative(self) -> Union[int, float, sympy.Expr]:
    """The relative exponent between negative and positive exponents."""
    return value.canonicalize_half_turns(self.exponent_neg - self.exponent_pos)