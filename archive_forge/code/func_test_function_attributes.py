import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_function_attributes(self):
    func = self.function()
    func.args[0].add_attribute('zeroext')
    func.args[1].attributes.dereferenceable = 5
    func.args[1].attributes.dereferenceable_or_null = 10
    func.args[3].attributes.align = 4
    func.args[3].add_attribute('nonnull')
    func.return_value.add_attribute('noalias')
    asm = self.descr(func).strip()
    self.assertEqual(asm, 'declare noalias i32 @"my_func"(i32 zeroext %".1", i32 dereferenceable(5) dereferenceable_or_null(10) %".2", double %".3", i32* nonnull align 4 %".4")')
    self.assert_pickle_correctly(func)