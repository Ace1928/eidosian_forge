from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
def create_workingtree(self, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False):
    """See ControlDir.create_workingtree."""
    if hardlink:
        warning("can't support hardlinked working trees in %r" % (self,))
    try:
        result = self.open_workingtree(recommend_upgrade=False)
    except NoSuchFile:
        result = self._init_workingtree()
    if revision_id is not None:
        if revision_id == _mod_revision.NULL_REVISION:
            result.set_parent_ids([])
        else:
            result.set_parent_ids([revision_id])
    return result