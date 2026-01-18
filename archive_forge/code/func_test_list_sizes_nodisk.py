import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, urlparse, parse_qsl
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ktucloud import KTUCloudNodeDriver
def test_list_sizes_nodisk(self):
    KTUCloudStackMockHttp.fixture_tag = 'nodisk'
    sizes = self.driver.list_sizes()
    self.assertEqual(2, len(sizes))
    check = False
    size = sizes[1]
    if size.id == KTUCloudNodeDriver.EMPTY_DISKOFFERINGID:
        check = True
    self.assertTrue(check)