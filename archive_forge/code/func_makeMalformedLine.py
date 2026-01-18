import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def makeMalformedLine(self, text):
    self.assertRaises(MalformedLine, parse_line, text)