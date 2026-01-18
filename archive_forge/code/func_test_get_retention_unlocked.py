from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_get_retention_unlocked(self):
    bucket_uri = self.CreateBucketWithRetentionPolicy(retention_period_in_seconds=_SECONDS_IN_DAY)
    stdout = self.RunGsUtil(['retention', 'get', suri(bucket_uri)], return_stdout=True)
    if self._use_gcloud_storage:
        self.assertNotRegexpMatches(stdout, 'isLocked \\: true')
        self.assertRegex(stdout, "retentionPeriod\\: \\'86400\\'")
        self.assertRegex(stdout, "effectiveTime\\: \\'.*\\'")
    else:
        self.assertRegex(stdout, 'Retention Policy \\(UNLOCKED\\):')
        self.assertRegex(stdout, 'Duration: 1 Day\\(s\\)')
        self.assertRegex(stdout, 'Effective Time: .* GMT')
    actual_retention_policy = self.json_api.GetBucket(bucket_uri.bucket_name, fields=['retentionPolicy']).retentionPolicy
    if self._use_gcloud_storage:
        expected_effective_time = datetime.datetime.fromisoformat(re.search("effectiveTime\\: \\'(.*)\\'", stdout).group(1))
        actual_effective_time = actual_retention_policy.effectiveTime
    else:
        expected_effective_time = self._ConvertTimeStringToSeconds(re.search('(?<=Time: )[\\w,: ]+', stdout).group())
        actual_effective_time = self.DateTimeToSeconds(actual_retention_policy.effectiveTime.replace(tzinfo=None))
    self.assertEqual(actual_effective_time, expected_effective_time)