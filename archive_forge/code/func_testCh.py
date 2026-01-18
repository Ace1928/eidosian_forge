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
def testCh(self):
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['label', 'ch', '-l', '%s:%s' % (KEY1, VALUE1), '-l', '%s:%s' % (KEY2, VALUE2), suri(bucket_uri)])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
        self.DoAssertItemsMatch(self._LabelDictFromXmlString(stdout), self._LabelDictFromXmlString(self._label_xml))
    _Check1()
    self.RunGsUtil(['label', 'ch', '-d', KEY1, '-l', 'new_key:new_value', '-d', 'nonexistent-key', suri(bucket_uri)])
    expected_dict = {KEY2: VALUE2, 'new_key': 'new_value'}

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
        self.DoAssertItemsMatch(self._LabelDictFromXmlString(stdout), expected_dict)
    _Check2()