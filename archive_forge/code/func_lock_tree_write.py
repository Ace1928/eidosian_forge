from io import BytesIO
from ... import conflicts as _mod_conflicts
from ... import errors, lock, osutils
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...bzr import conflicts as _mod_bzr_conflicts
from ...bzr import inventory
from ...bzr import transform as bzr_transform
from ...bzr import xml5
from ...bzr.workingtree_3 import PreDirStateWorkingTree
from ...mutabletree import MutableTree
from ...transport.local import LocalTransport
from ...workingtree import WorkingTreeFormat
def lock_tree_write(self):
    """See WorkingTree.lock_tree_write().

        In Format2 WorkingTrees we have a single lock for the branch and tree
        so lock_tree_write() degrades to lock_write().

        :return: An object with an unlock method which will release the lock
            obtained.
        """
    self.branch.lock_write()
    try:
        token = self._control_files.lock_write()
        return lock.LogicalLockResult(self.unlock, token)
    except:
        self.branch.unlock()
        raise