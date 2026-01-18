from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
def testLogging(self):
    """Tests enabling and disabling logging."""
    bucket_uri = self.CreateBucket()
    bucket_suri = suri(bucket_uri)
    stderr = self.RunGsUtil(self._enable_log_cmd + ['-b', bucket_suri, bucket_suri], return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('Updating', stderr)
    else:
        self.assertIn('Enabling logging', stderr)
    stdout = self.RunGsUtil(self._get_log_cmd + [bucket_suri], return_stdout=True)
    if self._use_gcloud_storage:
        _, _, prefixless_bucket = bucket_suri.partition('://')
        self.assertIn('"logBucket": "{}"'.format(prefixless_bucket), stdout)
        self.assertIn('"logObjectPrefix": "{}"'.format(prefixless_bucket), stdout)
    else:
        self.assertIn('LogObjectPrefix'.lower(), stdout.lower())
    stderr = self.RunGsUtil(self._disable_log_cmd + [bucket_suri], return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('Updating', stderr)
    else:
        self.assertIn('Disabling logging', stderr)