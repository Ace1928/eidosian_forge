import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_function_section(self):
    func = self.function()
    func.section = 'a_section'
    asm = self.descr(func).strip()
    self.assertEqual(asm, f'declare {self.proto} section "a_section"')
    self.assert_pickle_correctly(func)