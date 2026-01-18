import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_if_else_likely(self):

    def check(likely):
        block = self.block(name='one')
        builder = ir.IRBuilder(block)
        z = ir.Constant(int1, 0)
        with builder.if_else(z, likely=likely) as (then, otherwise):
            with then:
                builder.branch(block)
            with otherwise:
                builder.ret_void()
        self.check_func_body(builder.function, '                one:\n                    br i1 0, label %"one.if", label %"one.else", !prof !0\n                one.if:\n                    br label %"one"\n                one.else:\n                    ret void\n                one.endif:\n                ')
        return builder
    builder = check(True)
    self.check_metadata(builder.module, '            !0 = !{ !"branch_weights", i32 99, i32 1 }\n            ')
    builder = check(False)
    self.check_metadata(builder.module, '            !0 = !{ !"branch_weights", i32 1, i32 99 }\n            ')