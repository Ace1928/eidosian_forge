from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def test_convert_knit_merges_dirstate(self):
    tree = self.make_branch_and_tree('tree', format='knit')
    tree.commit('first post')
    merge_tree = tree.controldir.sprout('tree2').open_workingtree()
    rev_id2 = tree.commit('second post')
    rev_id3 = merge_tree.commit('second merge post')
    tree.merge_from_branch(merge_tree.branch)
    target = controldir.format_registry.make_controldir('dirstate')
    converter = tree.controldir._format.get_converter(target)
    converter.convert(tree.controldir, None)
    new_tree = workingtree.WorkingTree.open('tree')
    self.assertIs(new_tree.__class__, workingtree_4.WorkingTree4)
    self.assertEqual(rev_id2, new_tree.last_revision())
    self.assertEqual([rev_id2, rev_id3], new_tree.get_parent_ids())
    for path in ['basis-inventory-cache', 'inventory', 'last-revision', 'pending-merges', 'stat-cache']:
        self.assertPathDoesNotExist('tree/.bzr/checkout/' + path)