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
def test_invalid_partial_bucket_name(self):
    """Tests tab completion with a partial URL that by itself is not valid.

    The bucket name in a Cloud URL cannot end in a dash, but a partial URL
    during tab completion may end in a dash and completion should still work.
    """
    bucket_base_name = self.MakeTempName('bucket', prefix='aaa-')
    bucket_name = bucket_base_name + '-s'
    self.CreateBucket(bucket_name)
    request = '%s://%s-' % (self.default_provider, bucket_base_name)
    expected_result = '//%s/' % bucket_name
    self.RunGsUtilTabCompletion(['ls', request], expected_results=[expected_result])