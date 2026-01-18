import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def test_hammer_hashcache(self):
    hc = self.make_hashcache()
    for i in range(10000):
        with open('foo', 'wb') as f:
            last_content = b'%08x' % i
            f.write(last_content)
        last_sha1 = sha1(last_content)
        self.log('iteration %d: %r -> %r', i, last_content, last_sha1)
        got_sha1 = hc.get_sha1('foo')
        self.assertEqual(got_sha1, last_sha1)
        hc.write()
        hc = self.reopen_hashcache()