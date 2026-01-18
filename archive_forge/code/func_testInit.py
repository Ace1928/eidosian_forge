import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testInit(self):
    """Handle patches missing half the position, range tuple"""
    patchtext = b'--- orig/__vavg__.cl\n+++ mod/__vavg__.cl\n@@ -1 +1,2 @@\n __qbpsbezng__ = "erfgehpgherqgrkg ra"\n+__qbp__ = Na nygreangr Nepu pbzznaqyvar vagresnpr\n'
    self.compare_parsed(patchtext)