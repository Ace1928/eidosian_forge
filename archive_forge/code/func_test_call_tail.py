import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_call_tail(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    fun_ty = ir.FunctionType(ir.VoidType(), ())
    fun = ir.Function(builder.function.module, fun_ty, 'my_fun')
    builder.call(fun, ())
    builder.call(fun, (), tail=False)
    builder.call(fun, (), tail=True)
    builder.call(fun, (), tail='tail')
    builder.call(fun, (), tail='notail')
    builder.call(fun, (), tail='musttail')
    builder.call(fun, (), tail=[])
    builder.call(fun, (), tail='not a marker')
    self.check_block(block, '        my_block:\n            call void @"my_fun"()\n            call void @"my_fun"()\n            tail call void @"my_fun"()\n            tail call void @"my_fun"()\n            notail call void @"my_fun"()\n            musttail call void @"my_fun"()\n            call void @"my_fun"()\n            tail call void @"my_fun"()\n        ')