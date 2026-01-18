from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def assertStepOne(self, has_more, path, file_id, iterator):
    retval = multiwalker.MultiWalker._step_one(iterator)
    if not has_more:
        self.assertIs(None, path)
        self.assertIs(None, file_id)
        self.assertEqual((False, None, None), retval)
    else:
        self.assertEqual((has_more, path, file_id), (retval[0], retval[1], retval[2].file_id))