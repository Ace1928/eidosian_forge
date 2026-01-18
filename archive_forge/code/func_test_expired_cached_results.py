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
def test_expired_cached_results(self):
    """Tests tab completion results not returned from cache when too old."""
    with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
        bucket_base_name = self.MakeTempName('bucket')
        bucket_name = bucket_base_name + '-suffix'
        self.CreateBucket(bucket_name)
        request = '%s://%s' % (self.default_provider, bucket_base_name)
        expected_result = '%s://%s/' % (self.default_provider, bucket_name)
        cached_results = ['//%s1' % bucket_name, '//%s2' % bucket_name]
        _WriteTabCompletionCache(request, cached_results, time.time() - TAB_COMPLETE_CACHE_TTL)
        completer = CloudObjectCompleter(self.MakeGsUtilApi())
        results = completer(request)
        self.assertEqual([expected_result], results)