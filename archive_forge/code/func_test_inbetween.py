import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_inbetween(self):
    self.assertMultiBisectRight([(1, ['b'])], ['b'], ['a', 'c'])
    self.assertMultiBisectRight([(1, ['b', 'c', 'd']), (2, ['f', 'g'])], ['b', 'c', 'd', 'f', 'g'], ['a', 'e', 'h'])