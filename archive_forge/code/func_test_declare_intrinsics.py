import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_declare_intrinsics(self):
    module = self.module()
    pint8 = int8.as_pointer()
    powi = module.declare_intrinsic('llvm.powi', [dbl])
    memset = module.declare_intrinsic('llvm.memset', [pint8, int32])
    memcpy = module.declare_intrinsic('llvm.memcpy', [pint8, pint8, int32])
    assume = module.declare_intrinsic('llvm.assume')
    self.check_descr(self.descr(powi).strip(), '            declare double @"llvm.powi.f64"(double %".1", i32 %".2")')
    self.check_descr(self.descr(memset).strip(), '            declare void @"llvm.memset.p0i8.i32"(i8* %".1", i8 %".2", i32 %".3", i1 %".4")')
    self.check_descr(self.descr(memcpy).strip(), '            declare void @"llvm.memcpy.p0i8.p0i8.i32"(i8* %".1", i8* %".2", i32 %".3", i1 %".4")')
    self.check_descr(self.descr(assume).strip(), '            declare void @"llvm.assume"(i1 %".1")')