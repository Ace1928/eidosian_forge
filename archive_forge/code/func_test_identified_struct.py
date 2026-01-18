import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_identified_struct(self):
    context = ir.Context()
    mytype = context.get_identified_type('MyType')
    module = ir.Module(context=context)
    self.assertTrue(mytype.is_opaque)
    self.assert_valid_ir(module)
    oldstr = str(module)
    mytype.set_body(ir.IntType(32), ir.IntType(64), ir.FloatType())
    self.assertFalse(mytype.is_opaque)
    self.assert_valid_ir(module)
    self.assertNotEqual(oldstr, str(module))