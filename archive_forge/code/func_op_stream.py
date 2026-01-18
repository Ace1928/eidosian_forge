import sys
import dis
from typing import List, Tuple, TypeVar
from types import FunctionType
def op_stream(code, max):
    """Generator function: convert Python bytecode into a sequence of
    opcode-argument pairs."""
    i = [0]

    def next():
        val = code[i[0]]
        i[0] += 1
        return val
    ext_arg = 0
    while i[0] <= max:
        op, arg = (next(), next())
        if op == dis.EXTENDED_ARG:
            ext_arg += arg
            ext_arg <<= 8
            continue
        else:
            yield (op, arg + ext_arg)
            ext_arg = 0