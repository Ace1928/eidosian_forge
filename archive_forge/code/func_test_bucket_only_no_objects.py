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
def test_bucket_only_no_objects(self):
    """Tests that bucket-only tab completion doesn't match objects."""
    object_name = self.MakeTempName('obj')
    object_uri = self.CreateObject(object_name=object_name, contents=b'data')
    request = '%s://%s/%s' % (self.default_provider, object_uri.bucket_name, object_name[:-2])
    self.RunGsUtilTabCompletion(['rb', request], expected_results=[])