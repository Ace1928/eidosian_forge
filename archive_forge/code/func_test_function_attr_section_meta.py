import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_function_attr_section_meta(self):
    module = self.module()
    func = self.function(module)
    func.attributes.add('alwaysinline')
    func.section = 'a_section'
    func.set_metadata('dbg', module.add_metadata([]))
    asm = self.descr(func).strip()
    self.assertEqual(asm, f'declare {self.proto} alwaysinline section "a_section" !dbg !0')
    self.assert_pickle_correctly(func)