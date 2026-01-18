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
@SkipForXML(KMS_SKIP_MSG)
@SkipForS3(KMS_SKIP_MSG)
@Retry(AssertionError, tries=3, timeout_secs=1)
def test_create_with_k_flag_authorized(self):
    bucket_name = self.MakeTempName('bucket')
    bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
    key = self.GetKey(mutable=True)
    self.RunGsUtil(['kms', 'authorize', '-k', key])
    self.RunGsUtil(['mb', '-l', testcase.KmsTestingResources.KEYRING_LOCATION, '-k', key, suri(bucket_uri)], expected_status=0)