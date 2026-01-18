import os
import sys
import copy
import hmac
import math
import hashlib
import os.path  # pylint: disable-msg=W0404
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import StringIO, b, httplib, urlquote
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import MalformedResponseError
from libcloud.storage.base import CHUNK_SIZE, Object, Container
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.cloudfiles import CloudFilesStorageDriver
def test__upload_object_manifest(self):
    hash_function = self.driver._get_hash_function()
    hash_function.update(b(''))
    data_hash = hash_function.hexdigest()
    fake_response = type('CloudFilesResponse', (), {'headers': {'etag': data_hash}})
    _request = self.driver.connection.request
    mocked_request = mock.Mock(return_value=fake_response)
    self.driver.connection.request = mocked_request
    container = Container(name='foo_bar_container', extra={}, driver=self)
    object_name = 'test_object'
    self.driver._upload_object_manifest(container, object_name)
    func_args, func_kwargs = tuple(mocked_request.call_args)
    self.driver.connection.request = _request
    self.assertEqual(func_args[0], '/' + container.name + '/' + object_name)
    self.assertEqual(func_kwargs['headers']['X-Object-Manifest'], container.name + '/' + object_name + '/')
    self.assertEqual(func_kwargs['method'], 'PUT')