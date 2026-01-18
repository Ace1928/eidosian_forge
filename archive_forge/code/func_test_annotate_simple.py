from breezy.tests.per_tree import TestCaseWithTree
def test_annotate_simple(self):
    tree, revids = self.get_simple_tree()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual([(revids[1], b'second\n'), (revids[0], b'content\n')], list(tree.annotate_iter('one')))