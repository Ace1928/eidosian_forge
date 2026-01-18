import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_builder_load_reg(self):
    mod = self.module()
    foo = ir.Function(mod, ir.FunctionType(ir.VoidType(), []), 'foo')
    builder = ir.IRBuilder(foo.append_basic_block(''))
    builder.load_reg(ir.IntType(64), 'rax')
    builder.ret_void()
    pat = 'call i64 asm "", "={rax}"'
    self.assertInText(pat, str(mod))
    self.assert_valid_ir(mod)