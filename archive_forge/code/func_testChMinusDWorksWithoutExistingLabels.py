from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import xml
from xml.dom.minidom import parseString
from xml.sax import _exceptions as SaxExceptions
import six
import boto
from boto import handler
from boto.s3.tagging import Tags
from gslib.exception import CommandException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retry_util import Retry
from gslib.utils.constants import UTF8
def testChMinusDWorksWithoutExistingLabels(self):
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['label', 'ch', '-d', 'dummy-key', suri(bucket_uri)])
    stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
    if self._use_gcloud_storage:
        self.assertIn('[]', stdout)
    else:
        self.assertIn('%s/ has no label configuration.' % suri(bucket_uri), stdout)