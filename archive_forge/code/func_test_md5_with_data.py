import hashlib
import hmac
from oslotest import base as test_base
import testscenarios
from oslo_utils import secretutils
def test_md5_with_data(self):
    digest = secretutils.md5(self._test_data).digest()
    self.assertEqual(digest, self._md5_digest)
    digest = secretutils.md5(self._test_data, usedforsecurity=True).digest()
    self.assertEqual(digest, self._md5_digest)
    digest = secretutils.md5(self._test_data, usedforsecurity=False).digest()
    self.assertEqual(digest, self._md5_digest)