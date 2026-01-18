import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testParsePatches(self):
    """Make sure file names can be extracted from tricky unified diffs"""
    patchtext = b'--- orig-7\n+++ mod-7\n@@ -1,10 +1,10 @@\n -- a\n--- b\n+++ c\n xx d\n xx e\n ++ f\n-++ g\n+-- h\n xx i\n xx j\n -- k\n--- l\n+++ m\n--- orig-8\n+++ mod-8\n@@ -1 +1 @@\n--- A\n+++ B\n@@ -1 +1 @@\n--- C\n+++ D\n'
    filenames = [(b'orig-7', b'mod-7'), (b'orig-8', b'mod-8')]
    patches = parse_patches(patchtext.splitlines(True))
    patch_files = []
    for patch in patches:
        patch_files.append((patch.oldname, patch.newname))
    self.assertEqual(patch_files, filenames)