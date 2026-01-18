import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_invoke_attributes(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    fun_ty = ir.FunctionType(ir.VoidType(), (int32.as_pointer(), int32, int32.as_pointer()))
    fun = ir.Function(builder.function.module, fun_ty, 'fun')
    fun.calling_convention = 'fastcc'
    fun.args[0].add_attribute('sret')
    retval = builder.alloca(int32, name='retval')
    other = builder.alloca(int32, name='other')
    bb_normal = builder.function.append_basic_block(name='normal')
    bb_unwind = builder.function.append_basic_block(name='unwind')
    builder.invoke(fun, (retval, ir.Constant(int32, 42), other), bb_normal, bb_unwind, cconv='fastcc', fastmath='fast', attrs='noinline', arg_attrs={0: ('sret', 'noalias'), 2: 'noalias'})
    self.check_block_regex(block, '        my_block:\n            %"retval" = alloca i32\n            %"other" = alloca i32\n            invoke fast fastcc void @"fun"\\(i32\\* noalias sret(\\(i32\\))? %"retval", i32 42, i32\\* noalias %"other"\\) noinline\n                to label %"normal" unwind label %"unwind"\n        ')