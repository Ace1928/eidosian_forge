import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_iter_key_prefix_2_key_element_refs(self):
    index = self.make_index(1, key_elements=2, nodes=[((b'name', b'fin1'), b'data', ([(b'ref', b'erence')],)), ((b'name', b'fin2'), b'beta', ([],)), ((b'ref', b'erence'), b'refdata', ([],))])
    self.assertEqual({(index, (b'name', b'fin1'), b'data', (((b'ref', b'erence'),),)), (index, (b'ref', b'erence'), b'refdata', ((),))}, set(index.iter_entries_prefix([(b'name', b'fin1'), (b'ref', b'erence')])))
    self.assertEqual({(index, (b'name', b'fin1'), b'data', (((b'ref', b'erence'),),)), (index, (b'name', b'fin2'), b'beta', ((),))}, set(index.iter_entries_prefix([(b'name', None)])))