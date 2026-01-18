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
def delta_on_metadata(self):
    """Return True if delta's are permitted on metadata streams.

        That is on revisions and signatures.
        """
    src_serializer = self.from_repository._format._serializer
    target_serializer = self.to_format._serializer
    return self.to_format._fetch_uses_deltas and src_serializer == target_serializer