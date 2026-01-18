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
def testInvalidCSEKConfigurationRaises(self):
    invalid_key = 'aP7KbmxLqDw1SWHeKvlfKOVgNRNNZc8L2sFz8ybLN==='
    with self.assertRaises(CommandException) as cm:
        CryptoKeyWrapperFromKey(invalid_key)
    self.assertIn('Configured encryption_key or decryption_key looked like a CSEK', cm.exception.reason)