import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test__find_layer_start_and_stop(self):
    index = self.make_1000_node_index()
    self.assertEqual((0, 1), index._find_layer_first_and_end(0))
    self.assertEqual((1, 10), index._find_layer_first_and_end(1))
    self.assertEqual((1, 10), index._find_layer_first_and_end(9))
    self.assertEqual((10, 1000), index._find_layer_first_and_end(10))
    self.assertEqual((10, 1000), index._find_layer_first_and_end(99))
    self.assertEqual((10, 1000), index._find_layer_first_and_end(999))