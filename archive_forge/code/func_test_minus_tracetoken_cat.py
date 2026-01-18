from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
def test_minus_tracetoken_cat(self):
    """Tests cat command with trace-token option."""
    key_uri = self.CreateObject(contents=b'0123456789')
    _, stderr = self.RunGsUtil(['-D', '--trace-token=THISISATOKEN', 'cat', suri(key_uri)], return_stdout=True, return_stderr=True)
    if self.test_api == ApiSelector.JSON:
        self.assertIn('You are running gsutil with trace output enabled.', stderr)
        self.assertRegex(stderr, '.*GET.*b/%s/o/%s\\?.*trace=token%%3ATHISISATOKEN' % (key_uri.bucket_name, key_uri.object_name))