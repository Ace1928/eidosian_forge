import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def test_hashcache_raise(self):
    """check that hashcache can raise BzrError"""
    self.requireFeature(OsFifoFeature)
    hc = self.make_hashcache()
    os.mkfifo('a')
    self.assertRaises(BzrError, hc.get_sha1, 'a')