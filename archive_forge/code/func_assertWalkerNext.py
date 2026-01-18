from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def assertWalkerNext(self, exp_path, exp_file_id, master_has_node, exp_other_paths, iterator):
    """Check what happens when we step the iterator.

        :param path: The path for this entry
        :param file_id: The file_id for this entry
        :param master_has_node: Does the master tree have this entry?
        :param exp_other_paths: A list of other_path values.
        :param iterator: The iterator to step
        """
    path, file_id, master_ie, other_values = next(iterator)
    self.assertEqual((exp_path, exp_file_id), (path, file_id), 'Master entry did not match')
    if master_has_node:
        self.assertIsNot(None, master_ie, 'master should have an entry')
    else:
        self.assertIs(None, master_ie, 'master should not have an entry')
    self.assertEqual(len(exp_other_paths), len(other_values), 'Wrong number of other entries')
    other_paths = []
    other_file_ids = []
    for path, ie in other_values:
        other_paths.append(path)
        if ie is None:
            other_file_ids.append(None)
        else:
            other_file_ids.append(ie.file_id)
    exp_file_ids = []
    for path in exp_other_paths:
        if path is None:
            exp_file_ids.append(None)
        else:
            exp_file_ids.append(file_id)
    self.assertEqual(exp_other_paths, other_paths, 'Other paths incorrect')
    self.assertEqual(exp_file_ids, other_file_ids, 'Other file_ids incorrect')