from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
@TestCase.run_test_in_subprocess(envvars=_NUMBA_OPT_0_ENV)
def test_DILocation_entry_blk(self):

    @njit(debug=True)
    def foo(a):
        return a + 1
    foo(123)
    full_ir = foo.inspect_llvm(foo.signatures[0])
    module = llvm.parse_assembly(full_ir)
    name = foo.overloads[foo.signatures[0]].fndesc.mangled_name
    funcs = [x for x in module.functions if x.name == name]
    self.assertEqual(len(funcs), 1)
    func = funcs[0]
    blocks = [x for x in func.blocks]
    self.assertEqual(len(blocks), 2)
    entry_block, body_block = blocks
    entry_instr = [x for x in entry_block.instructions]
    ujmp = entry_instr[-1]
    self.assertEqual(ujmp.opcode, 'br')
    ujmp_operands = [x for x in ujmp.operands]
    self.assertEqual(len(ujmp_operands), 1)
    target_data = ujmp_operands[0]
    target = str(target_data).split(':')[0].strip()
    self.assertEqual(target, body_block.name)
    self.assertTrue(str(ujmp).endswith(target))