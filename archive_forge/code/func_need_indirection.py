import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def need_indirection(type):
    return isinstance(type, model.StructOrUnion) or (isinstance(type, model.PrimitiveType) and type.is_complex_type())