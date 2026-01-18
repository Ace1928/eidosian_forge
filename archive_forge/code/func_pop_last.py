import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
def pop_last(self):
    """Call this if you want to 'revoke' the last compression.

        After this, the data structures will be rolled back, but you cannot do
        more compression.
        """
    self._delta_index = None
    del self.chunks[self._last[0]:]
    self.endpoint = self._last[1]
    self._last = None