import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testPDiff(self):
    """Parse a hunk header produced by diff -p"""
    header = b'@@ -407,7 +292,7 @@ bzr 0.18rc1  2007-07-10\n'
    hunk = hunk_from_header(header)
    self.assertEqual(b'bzr 0.18rc1  2007-07-10', hunk.tail)
    self.assertEqual(header, hunk.as_bytes())