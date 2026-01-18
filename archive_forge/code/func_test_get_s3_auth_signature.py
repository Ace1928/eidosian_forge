import re
import sys
import copy
import json
import unittest
import email.utils
from io import BytesIO
from unittest import mock
from unittest.mock import Mock, PropertyMock
import pytest
from libcloud.test import StorageMockHttp
from libcloud.utils.py3 import StringIO, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_GOOGLE_STORAGE_PARAMS
from libcloud.common.google import GoogleAuthType
from libcloud.storage.drivers import google_storage
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.test.common.test_google import GoogleTestCase
@mock.patch('libcloud.storage.drivers.s3.BaseS3Connection.get_auth_signature')
def test_get_s3_auth_signature(self, mock_s3_auth_sig_method):
    mock_s3_auth_sig_method.return_value = 'mock signature!'
    starting_params = {}
    starting_headers = {'Date': TODAY, 'x-goog-foo': 'X-GOOG: MAINTAIN UPPERCASE!', 'x-Goog-bar': 'Header key should be lowered', 'Content-Type': 'application/mIXED casING MAINTAINED', 'Other': 'LOWER THIS!'}
    modified_headers = {'date': TODAY, 'content-type': 'application/mIXED casING MAINTAINED', 'x-goog-foo': 'X-GOOG: MAINTAIN UPPERCASE!', 'x-goog-bar': 'Header key should be lowered', 'other': 'lower this!'}
    conn = CONN_CLS('foo_user', 'bar_key', secure=True, auth_type=GoogleAuthType.GCS_S3)
    conn.method = 'GET'
    conn.action = '/path'
    result = conn._get_s3_auth_signature(starting_params, starting_headers)
    self.assertNotEqual(starting_headers, modified_headers)
    self.assertEqual(result, 'mock signature!')
    mock_s3_auth_sig_method.assert_called_once_with(method='GET', headers=modified_headers, params=starting_params, expires=None, secret_key='bar_key', path='/path', vendor_prefix='x-goog')