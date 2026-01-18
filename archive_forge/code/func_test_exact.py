import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_exact(self):
    self.assertMultiBisectRight([(1, ['a'])], ['a'], ['a'])
    self.assertMultiBisectRight([(1, ['a']), (2, ['b'])], ['a', 'b'], ['a', 'b'])
    self.assertMultiBisectRight([(1, ['a']), (3, ['c'])], ['a', 'c'], ['a', 'b', 'c'])