import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def set_cached_offsets(self, index, cached_offsets):
    """Monkeypatch to give a canned answer for _get_offsets_for...()."""

    def _get_offsets_to_cached_pages():
        cached = set(cached_offsets)
        return cached
    index._get_offsets_to_cached_pages = _get_offsets_to_cached_pages