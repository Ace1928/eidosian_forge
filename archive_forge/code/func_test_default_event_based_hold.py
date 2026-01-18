from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_default_event_based_hold(self):
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['retention', 'event-default', 'set', suri(bucket_uri)])
    self._VerifyDefaultEventBasedHold(bucket_uri, expected_default_event_based_hold=True)
    self.RunGsUtil(['retention', 'event-default', 'release', suri(bucket_uri)])
    self._VerifyDefaultEventBasedHold(bucket_uri, expected_default_event_based_hold=False)