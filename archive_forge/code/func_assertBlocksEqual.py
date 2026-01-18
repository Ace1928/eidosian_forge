import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
def assertBlocksEqual(self, code, *expected_blocks):
    self.assertEqual(len(code), len(expected_blocks))
    for block1, block2 in zip(code, expected_blocks):
        block_index = code.get_block_index(block1)
        self.assertListEqual(list(block1), block2, 'Block #%s is different' % block_index)