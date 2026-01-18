import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_call_attributes(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    fun_ty = ir.FunctionType(ir.VoidType(), (int32.as_pointer(), int32, int32.as_pointer()))
    fun = ir.Function(builder.function.module, fun_ty, 'fun')
    fun.args[0].add_attribute('sret')
    retval = builder.alloca(int32, name='retval')
    other = builder.alloca(int32, name='other')
    builder.call(fun, (retval, ir.Constant(int32, 42), other), arg_attrs={0: ('sret', 'noalias'), 2: 'noalias'})
    self.check_block_regex(block, '        my_block:\n            %"retval" = alloca i32\n            %"other" = alloca i32\n            call void @"fun"\\(i32\\* noalias sret(\\(i32\\))? %"retval", i32 42, i32\\* noalias %"other"\\)\n        ')