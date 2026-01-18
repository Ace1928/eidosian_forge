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
def get_branch_transport(self, branch_format, name=None):
    """See BzrDir.get_branch_transport()."""
    if name:
        raise NoColocatedBranchSupport(self)
    if branch_format is None:
        return self.transport
    try:
        branch_format.get_format_string()
    except NotImplementedError:
        return self.transport
    raise errors.IncompatibleFormat(branch_format, self._format)