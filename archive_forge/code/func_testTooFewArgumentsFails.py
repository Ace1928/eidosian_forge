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
def testTooFewArgumentsFails(self):
    """Ensures label commands fail with too few arguments."""
    invocations_missing_args = (['label'], ['label', 'set'], ['label', 'set', 'filename'], ['label', 'get'], ['label', 'ch'], ['label', 'ch', '-l', 'key:val'])
    for arg_list in invocations_missing_args:
        stderr = self.RunGsUtil(arg_list, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
    stderr = self.RunGsUtil(['label', 'ch', 'gs://some-nonexistent-foobar-bucket-name'], return_stderr=True, expected_status=1)
    self.assertIn('Please specify at least one label change', stderr)