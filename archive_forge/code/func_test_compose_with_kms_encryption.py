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
@SkipForS3('Test uses gs-specific KMS encryption')
def test_compose_with_kms_encryption(self):
    """Tests composing encrypted objects."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('gsutil does not support encryption with the XML API')
    bucket_uri = self.CreateBucket()
    object_uri1 = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
    object_uri2 = self.CreateObject(bucket_uri=bucket_uri, contents=b'bar')
    obj_suri = suri(bucket_uri, 'composed')
    key_fqn = AuthorizeProjectToUseTestingKmsKey()
    with SetBotoConfigForTest([('GSUtil', 'encryption_key', key_fqn)]):
        self.RunGsUtil(['compose', suri(object_uri1), suri(object_uri2), obj_suri])
    with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
        self.AssertObjectUsesCMEK(obj_suri, key_fqn)