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
@SkipForS3('S3 returns success when bucket already exists.')
def test_mb_bucket_exists(self):
    bucket_uri = self.CreateBucket()
    stderr = self.RunGsUtil(['mb', suri(bucket_uri)], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('HTTPError 409: The requested bucket name is not available.', stderr)
    else:
        self.assertIn('already exists', stderr)