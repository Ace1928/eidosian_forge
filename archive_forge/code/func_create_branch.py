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
def create_branch(self, name=None, repository=None, append_revisions_only=None):
    """See ControlDir.create_branch."""
    if repository is not None:
        raise NotImplementedError('create_branch(repository=<not None>) on {!r}'.format(self))
    return self._format.get_branch_format().initialize(self, name=name, append_revisions_only=append_revisions_only)