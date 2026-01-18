from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_lock_retention_userDoesNotConfirm(self):
    bucket_uri = self.CreateBucketWithRetentionPolicy(retention_period_in_seconds=_SECONDS_IN_DAY)
    stderr = self.RunGsUtil(['retention', 'lock', suri(bucket_uri)], stdin='n', return_stderr=True)
    self.assertRegex(stderr, 'Abort [Ll]ocking [Rr]etention [Pp]olicy on')
    self.VerifyRetentionPolicy(bucket_uri, expected_retention_period_in_seconds=_SECONDS_IN_DAY)