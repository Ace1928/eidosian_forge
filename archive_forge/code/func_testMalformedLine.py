import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testMalformedLine(self):
    """Parse invalid valid hunk lines"""
    self.makeMalformedLine(b'hello\n')