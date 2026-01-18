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
def testSetOverwritesOldLabelConfig(self):
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['label', 'set', self.json_fpath, suri(bucket_uri)])
    new_key_1 = 'new_key_1'
    new_key_2 = 'new_key_2'
    new_value_1 = 'new_value_1'
    new_value_2 = 'new_value_2'
    new_json = {new_key_1: new_value_1, new_key_2: new_value_2, KEY1: 'different_value_for_an_existing_key'}
    new_json_fpath = self.CreateTempFile(contents=json.dumps(new_json).encode('ascii'))
    self.RunGsUtil(['label', 'set', new_json_fpath, suri(bucket_uri)])
    stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
    self.assertDictEqual(json.loads(stdout), new_json)