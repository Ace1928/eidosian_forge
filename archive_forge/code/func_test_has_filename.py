from ....tests import TestCase, TestCaseWithTransport
from ....treebuilder import TreeBuilder
from ..maptree import MapTree, map_file_ids
def test_has_filename(self):
    self.oldtree.lock_write()
    builder = TreeBuilder()
    builder.start_tree(self.oldtree)
    builder.build(['foo'])
    builder.finish_tree()
    self.maptree = MapTree(self.oldtree, {})
    self.oldtree.unlock()
    self.assertTrue(self.maptree.has_filename('foo'))
    self.assertTrue(self.oldtree.has_filename('foo'))
    self.assertFalse(self.maptree.has_filename('bar'))