import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_if_then(self):
    block = self.block(name='one')
    builder = ir.IRBuilder(block)
    z = ir.Constant(int1, 0)
    a = builder.add(z, z, 'a')
    with builder.if_then(a) as bbend:
        builder.add(z, z, 'b')
    self.assertIs(builder.block, bbend)
    c = builder.add(z, z, 'c')
    with builder.if_then(c):
        builder.add(z, z, 'd')
        builder.branch(block)
    self.check_func_body(builder.function, '            one:\n                %"a" = add i1 0, 0\n                br i1 %"a", label %"one.if", label %"one.endif"\n            one.if:\n                %"b" = add i1 0, 0\n                br label %"one.endif"\n            one.endif:\n                %"c" = add i1 0, 0\n                br i1 %"c", label %"one.endif.if", label %"one.endif.endif"\n            one.endif.if:\n                %"d" = add i1 0, 0\n                br label %"one"\n            one.endif.endif:\n            ')