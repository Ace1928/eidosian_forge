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
def test__upload_object_part(self):
    _put_object = CloudFilesStorageDriver._put_object
    mocked__put_object = mock.Mock(return_value='test')
    CloudFilesStorageDriver._put_object = mocked__put_object
    part_number = 7
    object_name = 'test_object'
    expected_name = object_name + '/%08d' % part_number
    container = Container(name='foo_bar_container', extra={}, driver=self)
    self.driver._upload_object_part(container, object_name, part_number, None)
    CloudFilesStorageDriver._put_object = _put_object
    func_kwargs = tuple(mocked__put_object.call_args)[1]
    self.assertEqual(func_kwargs['object_name'], expected_name)
    self.assertEqual(func_kwargs['container'], container)