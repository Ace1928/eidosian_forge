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
@mock.patch.object(sys.stderr, 'write', autospec=True)
def test_does_not_warn_if_supported_double_wildcard(self, mock_stderr):
    storage_url.StorageUrlFromString('**')
    storage_url.StorageUrlFromString('gs://bucket/**')
    storage_url.StorageUrlFromString('**' + os.sep)
    storage_url.StorageUrlFromString('gs://bucket/**/')
    storage_url.StorageUrlFromString(os.sep + '**')
    storage_url.StorageUrlFromString('gs://bucket//**')
    storage_url.StorageUrlFromString(os.sep + '**' + os.sep)
    mock_stderr.assert_not_called()