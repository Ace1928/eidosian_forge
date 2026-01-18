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
def check_n_ary_compose(self, num_components):
    """Tests composing num_components object."""
    bucket_uri = self.CreateBucket()
    data_list = [('data-%d,' % i).encode('ascii') for i in xrange(num_components)]
    components = [self.CreateObject(bucket_uri=bucket_uri, contents=data).uri for data in data_list]
    composite = self.StorageUriCloneReplaceName(bucket_uri, self.MakeTempName('obj'))
    self.RunGsUtil(['compose'] + components + [composite.uri])
    self.assertEqual(composite.get_contents_as_string(), b''.join(data_list))