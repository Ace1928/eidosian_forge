from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from random import randint
import boto
import gslib.tests.testcase as testcase
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retention_util import SECONDS_IN_DAY
from gslib.utils.retention_util import SECONDS_IN_MONTH
from gslib.utils.retention_util import SECONDS_IN_YEAR
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
def test_create_with_retention_on_s3_urls_fails(self):
    bucket_name = self.MakeTempName('bucket')
    bucket_uri = boto.storage_uri('s3://%s' % bucket_name.lower(), suppress_consec_slashes=False)
    stderr = self.RunGsUtil(['mb', '--retention', '1y', suri(bucket_uri)], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('Features disallowed for S3: Setting Retention Period', stderr)
    else:
        self.assertRegex(stderr, 'Retention policy can only be specified for GCS buckets.')