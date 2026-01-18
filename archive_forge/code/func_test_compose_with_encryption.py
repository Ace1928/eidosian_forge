from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from six.moves import xrange
from six.moves import range
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import unittest
from gslib.project_id import PopulateProjectId
def test_compose_with_encryption(self):
    """Tests composing encrypted objects."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('gsutil does not support encryption with the XML API')
    bucket_uri = self.CreateBucket()
    object_uri1 = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo', encryption_key=TEST_ENCRYPTION_KEY1)
    object_uri2 = self.CreateObject(bucket_uri=bucket_uri, contents=b'bar', encryption_key=TEST_ENCRYPTION_KEY1)
    stderr = self.RunGsUtil(['compose', suri(object_uri1), suri(object_uri2), suri(bucket_uri, 'obj')], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('Missing decryption key', stderr)
    else:
        self.assertIn('is encrypted by a customer-supplied encryption key', stderr)
    with SetBotoConfigForTest([('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY1)]):
        stderr = self.RunGsUtil(['compose', suri(object_uri1), suri(object_uri2), suri(bucket_uri, 'obj')], expected_status=1, return_stderr=True)
        self.assertIn('provided encryption key is incorrect', stderr)
    with SetBotoConfigForTest([('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]):
        self.RunGsUtil(['compose', suri(object_uri1), suri(object_uri2), suri(bucket_uri, 'obj')])