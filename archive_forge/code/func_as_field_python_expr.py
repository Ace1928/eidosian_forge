import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def as_field_python_expr(self):
    if self.field_type_op.op == OP_NOOP:
        size_expr = ''
    elif self.field_type_op.op == OP_BITFIELD:
        size_expr = format_four_bytes(self.fbitsize)
    else:
        raise NotImplementedError
    return "b'%s%s%s'" % (self.field_type_op.as_python_bytes(), size_expr, self.name)