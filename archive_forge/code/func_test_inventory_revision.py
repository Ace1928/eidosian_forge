import os
from breezy.tests import TestCaseWithTransport
def test_inventory_revision(self):
    self.build_tree(['b/d', 'e'])
    self.tree.add(['b/d', 'e'], ids=[b'd-id', b'e-id'])
    self.tree.commit('add files')
    self.tree.rename_one('b/d', 'd')
    self.tree.commit('rename b/d => d')
    self.assertInventoryEqual('a\nb\nb/c\n', '-r 1')
    self.assertInventoryEqual('a\nb\nb/c\nb/d\ne\n', '-r 2')
    self.assertInventoryEqual('b/d\n', '-r 2 b/d')
    self.assertInventoryEqual('b/d\n', '-r 2 d')
    self.tree.rename_one('e', 'b/e')
    self.tree.commit('rename e => b/e')
    self.assertInventoryEqual('b\nb/c\nb/d\ne\n', '-r 2 b')