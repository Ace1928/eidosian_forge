import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def test_hashcache_old_file(self):
    """An old file gives the right sha1 and hits"""
    hc = self.make_hashcache()
    hc.put_file('foo', b'hello')
    hc.pretend_to_sleep(20)
    self.assertEqual(hc.get_sha1('foo'), sha1(b'hello'))
    self.assertEqual(hc.miss_count, 1)
    self.assertEqual(hc.hit_count, 0)
    self.assertEqual(hc.get_sha1('foo'), sha1(b'hello'))
    self.assertEqual(hc.miss_count, 1)
    self.assertEqual(hc.hit_count, 1)
    hc.pretend_to_sleep(3)
    self.assertEqual(hc.get_sha1('foo'), sha1(b'hello'))
    self.assertEqual(hc.miss_count, 1)
    self.assertEqual(hc.hit_count, 2)