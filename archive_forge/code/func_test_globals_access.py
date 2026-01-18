import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_globals_access(self):
    mod = self.module()
    foo = ir.Function(mod, ir.FunctionType(ir.VoidType(), []), 'foo')
    ir.Function(mod, ir.FunctionType(ir.VoidType(), []), 'bar')
    globdouble = ir.GlobalVariable(mod, ir.DoubleType(), 'globdouble')
    self.assertEqual(mod.get_global('foo'), foo)
    self.assertEqual(mod.get_global('globdouble'), globdouble)
    with self.assertRaises(KeyError):
        mod.get_global('kkk')
    self.assertEqual(repr(globdouble), "<ir.GlobalVariable 'globdouble' of type 'double*'>")