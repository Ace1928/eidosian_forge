from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
def test_excludes(self):
    """Tests exclude pattern excluding certain file paths."""
    bucket_uri, obj_uris = self._create_nested_subdir()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check():
        stdout = self.RunGsUtil(['du', '-e', '*sub2/five*', '-e', '*sub1材/four', suri(bucket_uri)], return_stdout=True)
        self.assertSetEqual(set(stdout.splitlines()), set(['%-11s  %s' % (5, suri(obj_uris[0])), '%-11s  %s' % (4, suri(obj_uris[3])), '%-11s  %s/sub1材/sub2/' % (4, suri(bucket_uri)), '%-11s  %s/sub1材/' % (9, suri(bucket_uri))]))
    _Check()