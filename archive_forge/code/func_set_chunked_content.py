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
def set_chunked_content(self, content_chunks, length):
    """Set the content of this block to the given chunks."""
    self._content_length = length
    self._content_chunks = content_chunks
    self._content = None
    self._z_content_chunks = None