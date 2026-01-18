import os
import sys
import hmac
import base64
import tempfile
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files  # NOQA: F401
from libcloud.test import MockHttp  # pylint: disable-msg=E0611  # noqa
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import ET, StringIO, b, httplib, parse_qs, urlparse
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.storage.drivers.s3 import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
def test_get_object_no_content_type_and_etag_in_response_headers(self):
    self.mock_response_klass.type = 'get_object_no_content_type'
    obj = self.driver.get_object(container_name='test2', object_name='test')
    self.assertEqual(obj.name, 'test')
    self.assertEqual(obj.container.name, 'test2')
    self.assertEqual(obj.size, 12345)
    self.assertIsNone(obj.hash)
    self.assertEqual(obj.extra['last_modified'], 'Thu, 13 Sep 2012 07:13:22 GMT')
    self.assertTrue('etag' not in obj.extra)
    self.assertTrue('content_type' not in obj.extra)