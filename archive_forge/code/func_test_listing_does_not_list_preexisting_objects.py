from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import sys
import six
import boto
from gslib.commands.perfdiag import _GenerateFileData
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import unittest
from gslib.utils.system_util import IS_WINDOWS
from six import add_move, MovedModule
from six.moves import mock
def test_listing_does_not_list_preexisting_objects(self):
    test_objects = 1
    bucket_uri = self.CreateBucket()
    self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
    self.CreateObject(bucket_uri=bucket_uri, contents=b'bar')
    mock_log_handler = self.RunCommand('perfdiag', ['-n', str(test_objects), '-t', 'list', suri(bucket_uri)], return_log_handler=True)
    self.assertNotIn('Listing produced more than the expected %d object(s).' % test_objects, mock_log_handler.messages['warning'])