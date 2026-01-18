import re
import sys
import copy
import json
import unittest
import email.utils
from io import BytesIO
from unittest import mock
from unittest.mock import Mock, PropertyMock
import pytest
from libcloud.test import StorageMockHttp
from libcloud.utils.py3 import StringIO, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_GOOGLE_STORAGE_PARAMS
from libcloud.common.google import GoogleAuthType
from libcloud.storage.drivers import google_storage
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.test.common.test_google import GoogleTestCase
def test_download_object_data_is_not_buffered_in_memory(self):
    mock_response = Mock(name='mock response')
    mock_response.headers = {}
    mock_response.status_code = 200
    msg = '"content" attribute was accessed but it shouldn\'t have been'
    type(mock_response).content = PropertyMock(name='mock content attribute', side_effect=Exception(msg))
    mock_response.iter_content.return_value = StringIO('a' * 1000)
    self.driver.connection.connection.getresponse = Mock()
    self.driver.connection.connection.getresponse.return_value = mock_response
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    obj = Object(name='foo_bar_object_NO_BUFFER', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
    destination_path = self._file_path
    result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=True, delete_on_failure=True)
    self.assertTrue(result)