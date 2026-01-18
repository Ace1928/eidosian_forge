import ast
import _ast
from qiskit.utils.optionals import HAS_TWEEDLEDUM
from .exceptions import ClassicalFunctionParseError, ClassicalFunctionCompilerTypeError
def bit_binop(self, op, values):
    """Uses ClassicalFunctionVisitor.bitops to extend self._network"""
    bitop = ClassicalFunctionVisitor.bitops.get(type(op))
    if not bitop:
        raise ClassicalFunctionParseError('Unknown binop.op %s' % op)
    binop = getattr(self._network, bitop)
    left_type, left_signal = values[0]
    if left_type != 'Int1':
        raise ClassicalFunctionParseError('binop type error')
    for right_type, right_signal in values[1:]:
        if right_type != 'Int1':
            raise ClassicalFunctionParseError('binop type error')
        left_signal = binop(left_signal, right_signal)
    return ('Int1', left_signal)