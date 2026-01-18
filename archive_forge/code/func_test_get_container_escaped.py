import sys
import base64
import os.path
import unittest
import libcloud.utils.files
from libcloud.test import MockHttp, make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.atmos import AtmosDriver, AtmosConnection
from libcloud.storage.drivers.dummy import DummyIterator
def test_get_container_escaped(self):
    container = self.driver.get_container(container_name='test & container')
    self.assertEqual(container.name, 'test & container')
    self.assertEqual(container.extra['object_id'], 'b21cb59a2ba339d1afdd4810010b0a5aba2ab6b9')