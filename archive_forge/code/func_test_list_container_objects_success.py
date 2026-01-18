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
def test_list_container_objects_success(self):
    self.mock_response_klass.type = None
    AzureBlobsStorageDriver.RESPONSES_PER_REQUEST = 2
    container = Container(name='test_container', extra={}, driver=self.driver)
    objects = self.driver.list_container_objects(container=container)
    self.assertEqual(len(objects), 4)
    obj = objects[1]
    self.assertEqual(obj.name, 'object2.txt')
    self.assertEqual(obj.hash, '0x8CFB90F1BA8CD8F')
    self.assertEqual(obj.size, 1048576)
    self.assertEqual(obj.container.name, 'test_container')
    self.assertTrue('meta1' in obj.meta_data)
    self.assertTrue('meta2' in obj.meta_data)
    self.assertTrue('last_modified' in obj.extra)
    self.assertTrue('content_type' in obj.extra)
    self.assertTrue('content_encoding' in obj.extra)
    self.assertTrue('content_language' in obj.extra)