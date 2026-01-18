import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testAll(self):
    """Test parsing a whole patch"""
    with self.datafile('patchtext.patch') as f:
        patchtext = f.read()
    self.compare_parsed(patchtext)