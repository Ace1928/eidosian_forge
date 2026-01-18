import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testInvalidHeader(self):
    """Parse an invalid hunk header"""
    self.makeMalformed(b' -34,11 +50,6 \n')
    self.makeMalformed(b'@@ +50,6 -34,11 @@\n')
    self.makeMalformed(b'@@ -34,11 +50,6 @@')
    self.makeMalformed(b'@@ -34.5,11 +50,6 @@\n')
    self.makeMalformed(b'@@-34,11 +50,6@@\n')
    self.makeMalformed(b'@@ 34,11 50,6 @@\n')
    self.makeMalformed(b'@@ -34,11 @@\n')
    self.makeMalformed(b'@@ -34,11 +50,6.5 @@\n')
    self.makeMalformed(b'@@ -34,11 +50,-6 @@\n')