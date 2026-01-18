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
def test_download_object_success(self):
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
    destination_path = os.path.abspath(__file__) + '.temp'
    result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
    self.assertTrue(result)