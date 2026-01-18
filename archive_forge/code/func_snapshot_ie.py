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
def snapshot_ie(self, previous_revisions, ie, w, rev_id):
    if len(previous_revisions) == 1:
        previous_ie = next(iter(previous_revisions.values()))
        if ie._unchanged(previous_ie):
            ie.revision = previous_ie.revision
            return
    if ie.has_text():
        with self.branch.repository._text_store.get(ie.text_id) as f:
            file_lines = f.readlines()
        w.add_lines(rev_id, previous_revisions, file_lines)
        self.text_count += 1
    else:
        w.add_lines(rev_id, previous_revisions, [])
    ie.revision = rev_id