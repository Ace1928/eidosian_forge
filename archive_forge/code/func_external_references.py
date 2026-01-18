from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def external_references(self, ref_list_num):
    if self._root_node is None:
        self._get_root_node()
    if ref_list_num + 1 > self.node_ref_lists:
        raise ValueError('No ref list %d, index has %d ref lists' % (ref_list_num, self.node_ref_lists))
    keys = set()
    refs = set()
    for node in self.iter_all_entries():
        keys.add(node[1])
        refs.update(node[3][ref_list_num])
    return refs - keys