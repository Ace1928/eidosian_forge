import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testValidLine(self):
    """Parse a valid hunk line"""
    self.lineThing(b' hello\n', ContextLine)
    self.lineThing(b'+hello\n', InsertLine)
    self.lineThing(b'-hello\n', RemoveLine)