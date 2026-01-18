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
def test_ex_get_object_temp_url_no_key_raises_key_error(self):
    self.driver.ex_get_meta_data = mock.Mock()
    self.driver.ex_get_meta_data.return_value = {'container_count': 1, 'object_count': 1, 'bytes_used': 1, 'temp_url_key': None}
    container = Container(name='foo_bar_container', extra={}, driver=self)
    obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self)
    self.assertRaises(KeyError, self.driver.ex_get_object_temp_url, obj, 'GET')