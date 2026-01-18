import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_named_metadata_2(self):
    mod = self.module()
    m0 = mod.add_metadata([int32(123)])
    mod.add_named_metadata('foo', m0)
    mod.add_named_metadata('foo', [int64(456)])
    mod.add_named_metadata('foo', ['kernel'])
    mod.add_named_metadata('bar', [])
    self.assert_ir_line('!foo = !{ !0, !1, !2 }', mod)
    self.assert_ir_line('!0 = !{ i32 123 }', mod)
    self.assert_ir_line('!1 = !{ i64 456 }', mod)
    self.assert_ir_line('!2 = !{ !"kernel" }', mod)
    self.assert_ir_line('!bar = !{ !3 }', mod)
    self.assert_ir_line('!3 = !{  }', mod)
    self.assert_valid_ir(mod)