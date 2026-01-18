from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_annotate_mod_from_parent(self):
    tree, revid = self.make_single_rev_tree()
    self.build_tree_contents([('tree/file', b'initial content\nnew content\n')])
    annotations = tree.annotate_iter('file')
    self.assertEqual([(revid, b'initial content\n'), (b'current:', b'new content\n')], annotations)