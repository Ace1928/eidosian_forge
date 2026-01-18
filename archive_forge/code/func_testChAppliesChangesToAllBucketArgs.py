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
def testChAppliesChangesToAllBucketArgs(self):
    bucket_suris = [suri(self.CreateBucket()), suri(self.CreateBucket())]
    ch_subargs = ['-l', '%s:%s' % (KEY1, VALUE1), '-l', '%s:%s' % (KEY2, VALUE2)]
    stderr = self.RunGsUtil(['label', 'ch'] + ch_subargs + bucket_suris, return_stderr=True)
    actual = set(stderr.splitlines())
    expected = set([_get_label_setting_output(self._use_gcloud_storage, bucket_suri) for bucket_suri in bucket_suris])
    if self._use_gcloud_storage:
        self.assertTrue(all([x in stderr for x in expected]))
    else:
        self.assertSetEqual(actual, expected)
    for bucket_suri in bucket_suris:
        stdout = self.RunGsUtil(['label', 'get', bucket_suri], return_stdout=True)
        self.assertDictEqual(json.loads(stdout), self._label_dict)