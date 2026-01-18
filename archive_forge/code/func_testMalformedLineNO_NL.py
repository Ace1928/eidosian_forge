import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testMalformedLineNO_NL(self):
    """Parse invalid '\\ No newline at end of file' in hunk lines"""
    self.makeMalformedLine(NO_NL)