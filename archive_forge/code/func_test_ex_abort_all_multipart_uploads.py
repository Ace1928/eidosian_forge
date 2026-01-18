import os
import sys
import unittest
from unittest import mock
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_OSS_PARAMS
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.oss import CHUNK_SIZE, OSSConnection, OSSStorageDriver
from libcloud.storage.drivers.dummy import DummyIterator
def test_ex_abort_all_multipart_uploads(self):
    if not self.driver.supports_multipart_upload:
        return
    self.mock_response_klass.type = 'list_multipart'
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    with mock.patch('libcloud.storage.drivers.oss.OSSStorageDriver._abort_multipart', autospec=True) as mock_abort:
        self.driver.ex_abort_all_multipart_uploads(container)
        self.assertEqual(3, mock_abort.call_count)