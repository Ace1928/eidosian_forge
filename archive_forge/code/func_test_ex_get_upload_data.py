import os
import sys
import json
import tempfile
from unittest import mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import b, httplib
from libcloud.utils.files import exhaust_iterator
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.backblaze_b2 import BackblazeB2StorageDriver
def test_ex_get_upload_data(self):
    container = self.driver.list_containers()[0]
    container_id = container.extra['id']
    data = self.driver.ex_get_upload_data(container_id=container_id)
    self.assertEqual(data['authorizationToken'], 'nope')
    self.assertEqual(data['bucketId'], '481c37de2e1ab3bf5e150710')
    self.assertEqual(data['uploadUrl'], 'https://podxxx.backblaze.com/b2api/v1/b2_upload_file/abcd/defg')