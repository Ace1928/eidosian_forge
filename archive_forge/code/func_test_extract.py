from .. import branch, errors
from . import TestCaseWithTransport
def test_extract(self):
    self.build_tree(['a/', 'a/b/', 'a/b/c', 'a/d'])
    wt = self.make_branch_and_tree('a', format='rich-root-pack')
    wt.add(['b', 'b/c', 'd'], ids=[b'b-id', b'c-id', b'd-id'])
    wt.commit('added files')
    b_wt = wt.extract('b')
    self.assertTrue(b_wt.is_versioned(''))
    if b_wt.supports_setting_file_ids():
        self.assertEqual(b'b-id', b_wt.path2id(''))
        self.assertEqual(b'c-id', b_wt.path2id('c'))
        self.assertEqual('c', b_wt.id2path(b'c-id'))
        self.assertRaises(errors.BzrError, wt.id2path, b'b-id')
    self.assertEqual(b_wt.basedir, wt.abspath('b'))
    self.assertEqual(wt.get_parent_ids(), b_wt.get_parent_ids())
    self.assertEqual(wt.branch.last_revision(), b_wt.branch.last_revision())