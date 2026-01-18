import unittest
from Cython.Compiler import PyrexTypes as pt
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.PyrexTypes import CFuncTypeArg
def cfunctype(*arg_types):
    return pt.CFuncType(pt.c_int_type, [CFuncTypeArg('name', arg_type, None) for arg_type in arg_types])