import unittest
import dis
import struct
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def get_extended_arg_load_const(self):
    """
        Get a function with a EXTENDED_ARG opcode before a LOAD_CONST opcode.
        """

    def f():
        x = 5
        return x
    b = bytearray(f.__code__.co_code)
    consts = f.__code__.co_consts
    bytecode_format = '<BB'
    consts = consts + (None,) * self.bytecode_len + (42,)
    if utils.PYVERSION >= (3, 11):
        offset = 2
    else:
        offset = 0
    packed_extend_arg = struct.pack(bytecode_format, dis.EXTENDED_ARG, 1)
    b[:] = b[:offset] + packed_extend_arg + b[offset:]
    f.__code__ = f.__code__.replace(co_code=bytes(b), co_consts=consts)
    return f