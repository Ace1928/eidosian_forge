from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def test__step_one(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a', 'tree/b/', 'tree/b/c'])
    tree.add(['a', 'b', 'b/c'], ids=[b'a-id', b'b-id', b'c-id'])
    iterator = tree.iter_entries_by_dir()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    root_id = tree.path2id('')
    self.assertStepOne(True, '', root_id, iterator)
    self.assertStepOne(True, 'a', b'a-id', iterator)
    self.assertStepOne(True, 'b', b'b-id', iterator)
    self.assertStepOne(True, 'b/c', b'c-id', iterator)
    self.assertStepOne(False, None, None, iterator)
    self.assertStepOne(False, None, None, iterator)