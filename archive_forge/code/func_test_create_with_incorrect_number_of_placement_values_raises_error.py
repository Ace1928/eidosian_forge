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
@SkipForXML('The --placement flag only works for GCS JSON API.')
def test_create_with_incorrect_number_of_placement_values_raises_error(self):
    bucket_name = self.MakeTempName('bucket')
    bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
    expected_status = 2 if self._use_gcloud_storage else 1
    stderr = self.RunGsUtil(['mb', '--placement', 'val1,val2,val3', suri(bucket_uri)], return_stderr=True, expected_status=expected_status)
    if self._use_gcloud_storage:
        self.assertIn('--placement: too many args', stderr)
    else:
        self.assertIn('CommandException: Please specify two regions separated by comma without space. Specified: val1,val2,val3', stderr)