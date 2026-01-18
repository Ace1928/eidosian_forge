from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_default_event_based_hold_multiple_sequential(self):
    bucket1_uri = self.CreateBucket()
    bucket2_uri = self.CreateBucket()
    self.RunGsUtil(['retention', 'event-default', 'set', suri(bucket1_uri), suri(bucket2_uri)])
    self._VerifyDefaultEventBasedHold(bucket1_uri, expected_default_event_based_hold=True)
    self._VerifyDefaultEventBasedHold(bucket2_uri, expected_default_event_based_hold=True)