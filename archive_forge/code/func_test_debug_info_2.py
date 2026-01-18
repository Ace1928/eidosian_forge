import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_debug_info_2(self):
    mod = self.module()
    di1 = mod.add_debug_info('DIFile', {'filename': 'foo', 'directory': 'bar'})
    di2 = mod.add_debug_info('DIFile', {'filename': 'foo', 'directory': 'bar'})
    di3 = mod.add_debug_info('DIFile', {'filename': 'bar', 'directory': 'foo'})
    di4 = mod.add_debug_info('DIFile', {'filename': 'foo', 'directory': 'bar'}, is_distinct=True)
    self.assertIs(di1, di2)
    self.assertEqual(len({di1, di2, di3, di4}), 3)
    strmod = str(mod)
    self.assert_ir_line('!0 = !DIFile(directory: "bar", filename: "foo")', strmod)
    self.assert_ir_line('!1 = !DIFile(directory: "foo", filename: "bar")', strmod)
    self.assert_ir_line('!2 = distinct !DIFile(directory: "bar", filename: "foo")', strmod)
    self.assert_valid_ir(mod)