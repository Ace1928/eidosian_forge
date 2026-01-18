from breezy.tests.per_tree import TestCaseWithTree
def get_tree_with_ghost(self):
    tree = self.make_branch_and_tree('tree')
    if not tree.branch.repository._format.supports_ghosts:
        self.skipTest('repository format does not support ghosts')
    self.build_tree_contents([('tree/one', b'first\ncontent\n')])
    tree.add(['one'])
    rev_1 = tree.commit('one')
    tree.set_parent_ids([rev_1, b'ghost-one'])
    self.build_tree_contents([('tree/one', b'second\ncontent\n')])
    rev_2 = tree.commit('two')
    return (self._convert_tree(tree), [rev_1, rev_2])