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
def test_list_container_objects_iterator_has_more(self):
    self.mock_response_klass.type = 'ITERATOR'
    container = Container(name='test_container', extra={}, driver=self.driver)
    objects = self.driver.list_container_objects(container=container)
    obj = [o for o in objects if o.name == '1.zip'][0]
    self.assertEqual(obj.hash, '4397da7a7649e8085de9916c240e8166')
    self.assertEqual(obj.size, 1234567)
    self.assertEqual(obj.container.name, 'test_container')
    self.assertTrue(obj in objects)
    self.assertEqual(len(objects), 5)