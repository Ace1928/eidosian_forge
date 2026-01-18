import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def prepare_index(self, index, node_ref_lists, key_length, key_count, row_lengths, cached_offsets):
    """Setup the BTreeGraphIndex with some pre-canned information."""
    index.node_ref_lists = node_ref_lists
    index._key_length = key_length
    index._key_count = key_count
    index._row_lengths = row_lengths
    index._compute_row_offsets()
    index._root_node = btree_index._InternalNode(b'internal\noffset=0\n')
    self.set_cached_offsets(index, cached_offsets)