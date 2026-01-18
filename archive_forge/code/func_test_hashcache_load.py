import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def test_hashcache_load(self):
    hc = self.make_hashcache()
    self.build_tree_contents([('foo', b'contents')])
    pause()
    self.assertEqual(hc.get_sha1('foo'), sha1(b'contents'))
    hc.write()
    hc = self.reopen_hashcache()
    self.assertEqual(hc.get_sha1('foo'), sha1(b'contents'))
    self.assertEqual(hc.hit_count, 1)