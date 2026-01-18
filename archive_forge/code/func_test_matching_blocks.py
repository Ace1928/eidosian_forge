import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_matching_blocks(self):
    self.assertDiffBlocks('', '', [])
    self.assertDiffBlocks([], [], [])
    self.assertDiffBlocks('abc', '', [])
    self.assertDiffBlocks('', 'abc', [])
    self.assertDiffBlocks('abcd', 'abcd', [(0, 0, 4)])
    self.assertDiffBlocks('abcd', 'abce', [(0, 0, 3)])
    self.assertDiffBlocks('eabc', 'abce', [(1, 0, 3)])
    self.assertDiffBlocks('eabce', 'abce', [(1, 0, 4)])
    self.assertDiffBlocks('abcde', 'abXde', [(0, 0, 2), (3, 3, 2)])
    self.assertDiffBlocks('abcde', 'abXYZde', [(0, 0, 2), (3, 5, 2)])
    self.assertDiffBlocks('abde', 'abXYZde', [(0, 0, 2), (2, 5, 2)])
    self.assertDiffBlocks('abcdefghijklmnop', 'abcdefxydefghijklmnop', [(0, 0, 6), (6, 11, 10)])
    self.assertDiffBlocks(['hello there\n', 'world\n', 'how are you today?\n'], ['hello there\n', 'how are you today?\n'], [(0, 0, 1), (2, 1, 1)])
    self.assertDiffBlocks('aBccDe', 'abccde', [(0, 0, 1), (5, 5, 1)])
    self.assertDiffBlocks('aBcDec', 'abcdec', [(0, 0, 1), (2, 2, 1), (4, 4, 2)])
    self.assertDiffBlocks('aBcdEcdFg', 'abcdecdfg', [(0, 0, 1), (8, 8, 1)])
    self.assertDiffBlocks('aBcdEeXcdFg', 'abcdecdfg', [(0, 0, 1), (2, 2, 2), (5, 4, 1), (7, 5, 2), (10, 8, 1)])
    self.assertDiffBlocks('abbabbXd', 'cabbabxd', [(7, 7, 1)])
    self.assertDiffBlocks('abbabbbb', 'cabbabbc', [])
    self.assertDiffBlocks('bbbbbbbb', 'cbbbbbbc', [])