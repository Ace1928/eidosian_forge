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
def test_get_object_escaped(self):
    obj = self.driver.get_object(container_name='test & container', object_name='test & object')
    self.assertEqual(obj.container.name, 'test & container')
    self.assertEqual(obj.size, 555)
    self.assertEqual(obj.hash, '6b21c4a111ac178feacf9ec9d0c71f17')
    self.assertEqual(obj.extra['object_id'], '322dce3763aadc41acc55ef47867b8d74e45c31d6643')
    self.assertEqual(obj.extra['last_modified'], 'Tue, 25 Jan 2011 22:01:49 GMT')
    self.assertEqual(obj.meta_data['foo-bar'], 'test 1')
    self.assertEqual(obj.meta_data['bar-foo'], 'test 2')