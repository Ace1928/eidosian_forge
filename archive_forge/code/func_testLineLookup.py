import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testLineLookup(self):
    """Make sure we can accurately look up mod line from orig"""
    patch = parse_patch(self.datafile('diff'))
    orig = list(self.datafile('orig'))
    mod = list(self.datafile('mod'))
    removals = []
    for i in range(len(orig)):
        mod_pos = patch.pos_in_mod(i)
        if mod_pos is None:
            removals.append(orig[i])
            continue
        self.assertEqual(mod[mod_pos], orig[i])
    rem_iter = removals.__iter__()
    for hunk in patch.hunks:
        for line in hunk.lines:
            if isinstance(line, RemoveLine):
                self.assertEqual(line.contents, next(rem_iter))
    self.assertRaises(StopIteration, next, rem_iter)