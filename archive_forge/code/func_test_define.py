import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_define(self):
    func = self.function()
    func.attributes.add('alwaysinline')
    block = func.append_basic_block('my_block')
    builder = ir.IRBuilder(block)
    builder.ret_void()
    asm = self.descr(func)
    self.check_descr(asm, '            define {proto} alwaysinline\n            {{\n            my_block:\n                ret void\n            }}\n            '.format(proto=self.proto))