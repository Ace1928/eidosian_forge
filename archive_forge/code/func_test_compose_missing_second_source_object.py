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
def test_compose_missing_second_source_object(self):
    bucket_uri = self.CreateBucket()
    object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
    stderr = self.RunGsUtil(['compose', suri(object_uri), suri(bucket_uri, 'nonexistent-obj'), suri(bucket_uri, 'valid-destination')], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('The following URLs matched no objects or files:', stderr)
    else:
        self.assertIn('NotFoundException', stderr)
        if self.test_api == ApiSelector.JSON:
            self.assertIn('One of the source objects does not exist', stderr)