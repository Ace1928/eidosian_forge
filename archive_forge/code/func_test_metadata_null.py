import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_metadata_null(self):
    mod = self.module()
    mod.add_metadata([int32.as_pointer()(None)])
    self.assert_ir_line('!0 = !{ i32* null }', mod)
    self.assert_valid_ir(mod)
    mod = self.module()
    mod.add_metadata([None, int32(123)])
    self.assert_ir_line('!0 = !{ null, i32 123 }', mod)
    self.assert_valid_ir(mod)