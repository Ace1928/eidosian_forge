import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_function_metadata(self):
    module = self.module()
    func = self.function(module)
    func.set_metadata('dbg', module.add_metadata([]))
    asm = self.descr(func).strip()
    self.assertEqual(asm, f'declare {self.proto} !dbg !0')
    self.assert_pickle_correctly(func)