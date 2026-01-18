import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def test_hashcache_new_file(self):
    hc = self.make_hashcache()
    self.build_tree_contents([('foo', b'goodbye')])
    self.assertEqual(hc.get_sha1('foo'), sha1(b'goodbye'))