import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def test_incoming_phi_blocks(self):
    mod = self.module(asm_phi_blocks)
    func = mod.get_function('foo')
    blocks = list(func.blocks)
    instructions = list(blocks[-1].instructions)
    self.assertTrue(instructions[0].is_instruction)
    self.assertEqual(instructions[0].opcode, 'phi')
    incoming_blocks = list(instructions[0].incoming_blocks)
    self.assertEqual(len(incoming_blocks), 2)
    self.assertTrue(incoming_blocks[0].is_block)
    self.assertTrue(incoming_blocks[1].is_block)
    self.assertEqual(incoming_blocks[0], blocks[-1])
    self.assertEqual(incoming_blocks[1], blocks[0])
    self.assertNotEqual(instructions[1].opcode, 'phi')
    with self.assertRaises(ValueError):
        instructions[1].incoming_blocks