import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_before(self):
    self.assertMultiBisectRight([(0, ['a'])], ['a'], ['b'])
    self.assertMultiBisectRight([(0, ['a', 'b', 'c', 'd'])], ['a', 'b', 'c', 'd'], ['e', 'f', 'g'])