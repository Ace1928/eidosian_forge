from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_event_based_hold_bucket_with_no_retention(self):
    bucket_uri = self.CreateBucket()
    object_uri = self.CreateObject(bucket_uri=bucket_uri, contents='content')
    self._VerifyObjectHoldAndRetentionStatus(bucket_uri, object_uri, event_based_hold=None)
    self.RunGsUtil(['retention', 'event', 'set', suri(object_uri)])
    self._VerifyObjectHoldAndRetentionStatus(bucket_uri, object_uri, event_based_hold=True)
    self.RunGsUtil(['retention', 'event', 'release', suri(object_uri)])
    self._VerifyObjectHoldAndRetentionStatus(bucket_uri, object_uri, event_based_hold=False)