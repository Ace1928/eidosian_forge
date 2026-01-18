from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def test_convert_knit_one_parent_dirstate(self):
    tree = self.make_branch_and_tree('tree', format='knit')
    rev_id = tree.commit('first post')
    target = controldir.format_registry.make_controldir('dirstate')
    converter = tree.controldir._format.get_converter(target)
    converter.convert(tree.controldir, None)
    new_tree = workingtree.WorkingTree.open('tree')
    self.assertIs(new_tree.__class__, workingtree_4.WorkingTree4)
    self.assertEqual(rev_id, new_tree.last_revision())
    for path in ['basis-inventory-cache', 'inventory', 'last-revision', 'pending-merges', 'stat-cache']:
        self.assertPathDoesNotExist('tree/.bzr/checkout/' + path)