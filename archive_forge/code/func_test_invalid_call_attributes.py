import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_invalid_call_attributes(self):
    block = self.block()
    builder = ir.IRBuilder(block)
    fun_ty = ir.FunctionType(ir.VoidType(), ())
    fun = ir.Function(builder.function.module, fun_ty, 'fun')
    with self.assertRaises(ValueError):
        builder.call(fun, (), arg_attrs={0: 'sret'})