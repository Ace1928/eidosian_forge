from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_annotate_same_as_parent(self):
    tree, revid = self.make_single_rev_tree()
    annotations = tree.annotate_iter('file')
    self.assertEqual([(revid, b'initial content\n')], annotations)