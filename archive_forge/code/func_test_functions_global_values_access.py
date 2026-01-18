import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_functions_global_values_access(self):
    """
        Accessing functions and global values through Module.functions
        and Module.global_values.
        """
    mod = self.module()
    fty = ir.FunctionType(ir.VoidType(), [])
    foo = ir.Function(mod, fty, 'foo')
    bar = ir.Function(mod, fty, 'bar')
    globdouble = ir.GlobalVariable(mod, ir.DoubleType(), 'globdouble')
    self.assertEqual(set(mod.functions), set((foo, bar)))
    self.assertEqual(set(mod.global_values), set((foo, bar, globdouble)))