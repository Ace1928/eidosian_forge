import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def test_fanout_raise_2(self):
    mod, stats = self.check(self.fanout_raise_2)
    self.assertEqual(stats.fanout_raise, 0)