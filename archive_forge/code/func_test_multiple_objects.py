from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import time
from gslib.command import CreateOrGetGsutilLogger
from gslib.tab_complete import CloudObjectCompleter
from gslib.tab_complete import TAB_COMPLETE_CACHE_TTL
from gslib.tab_complete import TabCompletionCache
import gslib.tests.testcase as testcase
from gslib.tests.util import ARGCOMPLETE_AVAILABLE
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.tests.util import WorkingDirectory
from gslib.utils.boto_util import GetTabCompletionCacheFilename
def test_multiple_objects(self):
    """Tests tab completion matching multiple objects."""
    bucket_uri = self.CreateBucket()
    object_base_name = self.MakeTempName('obj')
    object1_name = object_base_name + '-suffix1'
    self.CreateObject(bucket_uri=bucket_uri, object_name=object1_name, contents=b'data')
    object2_name = object_base_name + '-suffix2'
    self.CreateObject(bucket_uri=bucket_uri, object_name=object2_name, contents=b'data')
    request = '%s://%s/%s' % (self.default_provider, bucket_uri.bucket_name, object_base_name)
    expected_result1 = '//%s/%s' % (bucket_uri.bucket_name, object1_name)
    expected_result2 = '//%s/%s' % (bucket_uri.bucket_name, object2_name)
    self.RunGsUtilTabCompletion(['ls', request], expected_results=[expected_result1, expected_result2])