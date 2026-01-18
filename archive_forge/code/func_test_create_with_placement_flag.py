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
def test_create_with_placement_flag(self):
    bucket_name = self.MakeTempName('bucket')
    bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
    self.RunGsUtil(['mb', '--placement', 'us-central1,us-west1', suri(bucket_uri)])
    stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
    self.assertRegex(stdout, 'ocations:\\s*\\[\\s*.US-CENTRAL1.,\\s*.US-WEST1')