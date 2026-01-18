from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from gslib.exception import CommandException
from gslib import storage_url
from gslib.exception import InvalidUrlError
from gslib.tests.testcase import base
from unittest import mock
def test_is_file_url_string(self):
    self.assertTrue(storage_url.IsFileUrlString('abc'))
    self.assertTrue(storage_url.IsFileUrlString('file://abc'))
    self.assertFalse(storage_url.IsFileUrlString('gs://abc'))
    self.assertFalse(storage_url.IsFileUrlString('s3://abc'))