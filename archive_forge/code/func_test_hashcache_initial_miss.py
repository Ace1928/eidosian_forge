import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def test_hashcache_initial_miss(self):
    """Get correct hash from an empty hashcache"""
    hc = self.make_hashcache()
    self.build_tree_contents([('foo', b'hello')])
    self.assertEqual(hc.get_sha1('foo'), b'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d')
    self.assertEqual(hc.miss_count, 1)
    self.assertEqual(hc.hit_count, 0)