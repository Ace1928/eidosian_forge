from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
def check_file_version_parents(self, texts, progress_bar=None):
    """Check the parents stored in a versioned file are correct.

        It also detects file versions that are not referenced by their
        corresponding revision's inventory.

        :returns: A tuple of (wrong_parents, dangling_file_versions).
            wrong_parents is a dict mapping {revision_id: (stored_parents,
            correct_parents)} for each revision_id where the stored parents
            are not correct.  dangling_file_versions is a set of (file_id,
            revision_id) tuples for versions that are present in this versioned
            file, but not used by the corresponding inventory.
        """
    local_progress = None
    if progress_bar is None:
        local_progress = ui.ui_factory.nested_progress_bar()
        progress_bar = local_progress
    try:
        return self._check_file_version_parents(texts, progress_bar)
    finally:
        if local_progress:
            local_progress.finished()