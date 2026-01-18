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
def testInitialAndSubsequentCh(self):
    bucket_uri = self.CreateBucket()
    ch_subargs = ['-l', '%s:%s' % (KEY1, VALUE1), '-l', '%s:%s' % (KEY2, VALUE2)]
    stderr = self.RunGsUtil(['label', 'ch'] + ch_subargs + [suri(bucket_uri)], return_stderr=True)
    expected_output = _get_label_setting_output(self._use_gcloud_storage, suri(bucket_uri))
    if self._use_gcloud_storage:
        self.assertIn(expected_output, stderr)
    else:
        self.assertEqual(stderr.strip(), expected_output)
    stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
    self.assertDictEqual(json.loads(stdout), self._label_dict)
    new_key = 'new-key'
    new_value = 'new-value'
    self.RunGsUtil(['label', 'ch', '-l', '%s:%s' % (new_key, new_value), '-d', KEY2, suri(bucket_uri)])
    stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
    actual = json.loads(stdout)
    expected = {KEY1: VALUE1, new_key: new_value}
    self.assertDictEqual(actual, expected)