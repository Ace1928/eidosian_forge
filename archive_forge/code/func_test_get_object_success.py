import os
import sys
import json
import tempfile
from io import BytesIO
from libcloud.test import generate_random_data  # pylint: disable-msg=E0611
from libcloud.test import unittest
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_AZURE_BLOBS_PARAMS, STORAGE_AZURITE_BLOBS_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.azure_blobs import (
def test_get_object_success(self):
    self.mock_response_klass.type = None
    obj = self.driver.get_object(container_name='test_container200', object_name='test')
    self.assertEqual(obj.name, 'test')
    self.assertEqual(obj.container.name, 'test_container200')
    self.assertEqual(obj.size, 12345)
    self.assertEqual(obj.hash, '0x8CFB877BB56A6FB')
    self.assertEqual(obj.extra['last_modified'], 'Fri, 04 Jan 2013 09:48:06 GMT')
    self.assertEqual(obj.extra['content_type'], 'application/zip')
    self.assertEqual(obj.meta_data['rabbits'], 'monkeys')