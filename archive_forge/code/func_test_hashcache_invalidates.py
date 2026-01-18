import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def test_hashcache_invalidates(self):
    hc = self.make_hashcache()
    hc.put_file('foo', b'hello')
    hc.pretend_to_sleep(20)
    hc.get_sha1('foo')
    hc.put_file('foo', b'h1llo')
    self.assertEqual(hc.get_sha1('foo'), sha1(b'h1llo'))
    self.assertEqual(hc.miss_count, 2)
    self.assertEqual(hc.hit_count, 0)