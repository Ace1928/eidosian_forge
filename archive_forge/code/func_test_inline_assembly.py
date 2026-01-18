import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_inline_assembly(self):
    mod = self.module()
    foo = ir.Function(mod, ir.FunctionType(ir.VoidType(), []), 'foo')
    builder = ir.IRBuilder(foo.append_basic_block(''))
    asmty = ir.FunctionType(int32, [int32])
    asm = ir.InlineAsm(asmty, 'mov $1, $2', '=r,r', side_effect=True)
    builder.call(asm, [int32(123)])
    builder.ret_void()
    pat = 'call i32 asm sideeffect "mov $1, $2", "=r,r" ( i32 123 )'
    self.assertInText(pat, str(mod))
    self.assert_valid_ir(mod)