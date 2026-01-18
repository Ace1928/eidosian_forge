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
def test_compose_with_precondition(self):
    """Tests composing objects with a destination precondition."""
    bucket_uri = self.CreateVersionedBucket()
    k1_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data1')
    k2_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data2')
    g1 = k1_uri.generation
    gen_match_header = 'x-goog-if-generation-match:%s' % g1
    self.RunGsUtil(['-h', gen_match_header, 'compose', suri(k1_uri), suri(k2_uri), suri(k1_uri)])
    stderr = self.RunGsUtil(['-h', gen_match_header, 'compose', suri(k1_uri), suri(k2_uri), suri(k1_uri)], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertIn('At least one of the pre-conditions you specified did not hold', stderr)
    else:
        self.assertIn('PreconditionException', stderr)