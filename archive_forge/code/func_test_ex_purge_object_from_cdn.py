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
def test_ex_purge_object_from_cdn(self):
    CloudFilesMockHttp.type = 'PURGE_SUCCESS'
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    obj = Object(name='object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self)
    self.assertTrue(self.driver.ex_purge_object_from_cdn(obj=obj))