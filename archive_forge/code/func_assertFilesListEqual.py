from breezy import osutils
from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
def assertFilesListEqual(self, tree, expected, **kwargs):
    with tree.lock_read():
        if tree.supports_file_ids:
            actual = [(path, status, kind, ie.file_id) for path, status, kind, ie in tree.list_files(**kwargs)]
            expected = [(path, status, kind, tree.path2id(osutils.pathjoin(kwargs.get('from_dir', ''), path))) for path, status, kind in expected]
        else:
            actual = [(path, status, kind) for path, status, kind, ie in tree.list_files(**kwargs)]
            expected = [(path, status, kind) for path, status, kind in expected]
    self.assertEqual(expected, actual)