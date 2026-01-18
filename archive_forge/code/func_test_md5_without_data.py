import hashlib
import hmac
from oslotest import base as test_base
import testscenarios
from oslo_utils import secretutils
def test_md5_without_data(self):
    md5 = secretutils.md5()
    md5.update(self._test_data)
    digest = md5.digest()
    self.assertEqual(digest, self._md5_digest)
    md5 = secretutils.md5(usedforsecurity=True)
    md5.update(self._test_data)
    digest = md5.digest()
    self.assertEqual(digest, self._md5_digest)
    md5 = secretutils.md5(usedforsecurity=False)
    md5.update(self._test_data)
    digest = md5.digest()
    self.assertEqual(digest, self._md5_digest)