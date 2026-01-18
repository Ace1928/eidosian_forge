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
def testNonSequentialDecryptionKeys(self):
    """Tests a config file with non-sequential decryption key numbering."""
    keys = []
    for _ in range(3):
        try:
            keys.append(base64.encodebytes(os.urandom(32)).rstrip(b'\n'))
        except AttributeError:
            keys.append(base64.encodestring(os.urandom(32)).rstrip(b'\n'))
    boto_config = [('GSUtil', 'decryption_key4', keys[2]), ('GSUtil', 'decryption_key1', keys[0]), ('GSUtil', 'decryption_key2', keys[1])]
    with SetBotoConfigForTest(boto_config):
        self.assertIsNone(FindMatchingCSEKInBotoConfig(Base64Sha256FromBase64EncryptionKey(keys[2]), boto.config))
        self.assertIsNotNone(FindMatchingCSEKInBotoConfig(Base64Sha256FromBase64EncryptionKey(keys[0]), boto.config))
        self.assertIsNotNone(FindMatchingCSEKInBotoConfig(Base64Sha256FromBase64EncryptionKey(keys[1]), boto.config))