import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_after(self):
    self.assertMultiBisectRight([(1, ['b'])], ['b'], ['a'])
    self.assertMultiBisectRight([(3, ['e', 'f', 'g'])], ['e', 'f', 'g'], ['a', 'b', 'c'])