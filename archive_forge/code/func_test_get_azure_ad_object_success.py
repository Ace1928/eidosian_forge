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
def test_get_azure_ad_object_success(self):
    AzureBlobsActiveDirectoryConnection.conn_class = AzureBlobsMockHttp
    driver = self.driver_type(key='fakeaccount1', secret='DEKjfhdakkdjfhei~', tenant_id='77777777-7777-7777-7777-777777777777', identity='55555555-5555-5555-5555-555555555555', auth_type='azureAd', secure=True)
    self.mock_response_klass.type = None
    container = driver.get_container(container_name='test_container200')
    self.assertTrue(container.name, 'test_container200')
    self.assertTrue(container.extra['etag'], '0x8CFB877BB56A6FB')
    self.assertTrue(container.extra['last_modified'], 'Fri, 04 Jan 2013 09:48:06 GMT')
    self.assertTrue(container.extra['lease']['status'], 'unlocked')
    self.assertTrue(container.extra['lease']['state'], 'available')
    self.assertTrue(container.extra['meta_data']['meta1'], 'value1')