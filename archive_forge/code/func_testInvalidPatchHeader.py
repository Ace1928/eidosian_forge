import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testInvalidPatchHeader(self):
    """Parse an invalid patch header"""
    lines = b'-- orig/commands.py\n+++ mod/dommands.py'.split(b'\n')
    self.assertRaises(MalformedPatchHeader, get_patch_names, lines.__iter__())