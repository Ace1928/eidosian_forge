from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import six
import boto
from gslib.exception import CommandException
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.tests.util import SetBotoConfigForTest
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
def testMaxDecryptionKeys(self):
    """Tests a config file with the maximum number of decryption keys."""
    keys = []
    boto_101_key_config = []
    for i in range(1, 102):
        try:
            keys.append(base64.encodebytes(os.urandom(32)).rstrip(b'\n'))
        except AttributeError:
            keys.append(base64.encodestring(os.urandom(32)).rstrip(b'\n'))
        boto_101_key_config.append(('GSUtil', 'decryption_key%s' % i, keys[i - 1]))
    with SetBotoConfigForTest(boto_101_key_config):
        self.assertIsNotNone(FindMatchingCSEKInBotoConfig(Base64Sha256FromBase64EncryptionKey(keys[0]), boto.config))
        self.assertIsNotNone(FindMatchingCSEKInBotoConfig(Base64Sha256FromBase64EncryptionKey(keys[99]), boto.config))
        self.assertIsNone(FindMatchingCSEKInBotoConfig(Base64Sha256FromBase64EncryptionKey(keys[100]), boto.config))
    boto_100_key_config = list(boto_101_key_config)
    boto_100_key_config.pop()
    with SetBotoConfigForTest(boto_100_key_config):
        self.assertIsNotNone(FindMatchingCSEKInBotoConfig(Base64Sha256FromBase64EncryptionKey(keys[0]), boto.config))
        self.assertIsNotNone(FindMatchingCSEKInBotoConfig(Base64Sha256FromBase64EncryptionKey(keys[99]), boto.config))