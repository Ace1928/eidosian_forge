from breezy import errors, tests
from breezy.branch import format_registry
from breezy.bzr.remote import RemoteBranchFormat
from breezy.tests import test_server
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import memory
def create_tree_with_merge(self):
    """Create a branch with a simple ancestry.

        The graph should look like:
            digraph H {
                "1" -> "2" -> "3";
                "1" -> "1.1.1" -> "3";
            }

        Or in ASCII:
            1
            |            2 1.1.1
            |/
            3
        """
    revmap = {}
    tree = self.make_branch_and_memory_tree('tree')
    with tree.lock_write():
        tree.add('')
        revmap['1'] = tree.commit('first')
        revmap['1.1.1'] = tree.commit('second')
        tree.branch.set_last_revision_info(1, revmap['1'])
        tree.set_parent_ids([revmap['1']])
        revmap['2'] = tree.commit('alt-second')
        tree.set_parent_ids([revmap['2'], revmap['1.1.1']])
        revmap['3'] = tree.commit('third')
    return (tree, revmap)