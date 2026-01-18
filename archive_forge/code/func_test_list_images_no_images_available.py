import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, urlparse, parse_qsl
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ktucloud import KTUCloudNodeDriver
def test_list_images_no_images_available(self):
    KTUCloudStackMockHttp.fixture_tag = 'notemplates'
    images = self.driver.list_images()
    self.assertEqual(0, len(images))