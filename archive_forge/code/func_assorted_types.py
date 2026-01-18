import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def assorted_types(self):
    """
        A bunch of mutually unequal types
        """
    context = ir.Context()
    types = [ir.LabelType(), ir.VoidType(), ir.FunctionType(int1, (int8, int8)), ir.FunctionType(int1, (int8,)), ir.FunctionType(int1, (int8,), var_arg=True), ir.FunctionType(int8, (int8,)), int1, int8, int32, flt, dbl, ir.ArrayType(flt, 5), ir.ArrayType(dbl, 5), ir.ArrayType(dbl, 4), ir.LiteralStructType((int1, int8)), ir.LiteralStructType((int8, int1)), context.get_identified_type('MyType1'), context.get_identified_type('MyType2')]
    types += [ir.PointerType(tp) for tp in types if not isinstance(tp, (ir.VoidType, ir.LabelType))]
    return types