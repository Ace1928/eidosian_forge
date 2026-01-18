import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def make_decref(self, m):
    fnty = ir.FunctionType(ir.VoidType(), [ptr_ty])
    return ir.Function(m, fnty, name='NRT_decref')