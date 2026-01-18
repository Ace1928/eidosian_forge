from .. import errors, tests, uncommit
def make_linear_tree(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    try:
        self.build_tree(['tree/one'])
        tree.add('one')
        rev_id1 = tree.commit('one')
        self.build_tree(['tree/two'])
        tree.add('two')
        rev_id2 = tree.commit('two')
    finally:
        tree.unlock()
    return (tree, [rev_id1, rev_id2])