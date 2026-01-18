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
def test_compose_copies_type_and_encoding_from_first_object(self):
    bucket_uri = self.CreateBucket()
    object_uri1 = self.CreateObject(bucket_uri=bucket_uri, contents=b'1')
    object_uri2 = self.CreateObject(bucket_uri=bucket_uri, contents=b'2')
    composite = self.StorageUriCloneReplaceName(bucket_uri, self.MakeTempName('obj'))
    self.RunGsUtil(['setmeta', '-h', 'Content-Type:python-x', '-h', 'Content-Encoding:gzip', suri(object_uri1)])
    self.RunGsUtil(['compose', suri(object_uri1), suri(object_uri2), suri(composite)])
    stdout = self.RunGsUtil(['stat', suri(composite)], return_stdout=True)
    self.assertRegex(stdout, 'Content-Type:\\s+python-x')
    self.assertRegex(stdout, 'Content-Encoding:\\s+gzip')