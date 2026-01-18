from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from random import randint
from unittest import mock
from gslib.cloud_api import AccessDeniedException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
def testKmsEncryptionFlow(self):
    bucket_uri = self.CreateBucket()
    key_fqn = self.kms_api.CreateCryptoKey(self.keyring_fqn, testcase.KmsTestingResources.CONSTANT_KEY_NAME)
    encryption_get_cmd = ['kms', 'encryption', suri(bucket_uri)]
    stdout = self.RunGsUtil(encryption_get_cmd, return_stdout=True)
    self.assertIn('Bucket %s has no default encryption key' % suri(bucket_uri), stdout)
    stdout = self.RunGsUtil(['kms', 'encryption', '-k', key_fqn, suri(bucket_uri)], return_stdout=True)
    self.assertIn('Setting default KMS key for bucket %s...' % suri(bucket_uri), stdout)
    stdout = self.RunGsUtil(encryption_get_cmd, return_stdout=True)
    self.assertIn('Default encryption key for %s:\n%s' % (suri(bucket_uri), key_fqn), stdout)
    stdout = self.RunGsUtil(['kms', 'encryption', '-d', suri(bucket_uri)], return_stdout=True)
    self.assertIn('Clearing default encryption key for %s...' % suri(bucket_uri), stdout)
    stdout = self.RunGsUtil(encryption_get_cmd, return_stdout=True)
    self.assertIn('Bucket %s has no default encryption key' % suri(bucket_uri), stdout)