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
def conflicts(self):
    with self.lock_read():
        conflicts = _mod_conflicts.ConflictList()
        for conflicted in self._iter_conflicts():
            text = True
            try:
                if osutils.file_kind(self.abspath(conflicted)) != 'file':
                    text = False
            except _mod_transport.NoSuchFile:
                text = False
            if text is True:
                for suffix in ('.THIS', '.OTHER'):
                    try:
                        kind = osutils.file_kind(self.abspath(conflicted + suffix))
                        if kind != 'file':
                            text = False
                    except _mod_transport.NoSuchFile:
                        text = False
                    if text is False:
                        break
            ctype = {True: 'text conflict', False: 'contents conflict'}[text]
            conflicts.append(_mod_bzr_conflicts.Conflict.factory(ctype, path=conflicted, file_id=self.path2id(conflicted)))
        return conflicts