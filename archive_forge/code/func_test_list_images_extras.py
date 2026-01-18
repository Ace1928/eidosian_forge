import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
def test_list_images_extras(self):
    images = self.driver.list_images()
    extra = images[-1].extra
    self.assertEqual(extra['arch'], 'i686')
    self.assertFalse(extra['compatibility_mode'])
    self.assertEqual(extra['created_at'], '2012-01-22T05:36:24Z')
    self.assertTrue('description' in extra)
    self.assertEqual(extra['disk_size'], 671)
    self.assertFalse('min_ram' in extra)
    self.assertFalse(extra['official'])
    self.assertEqual(extra['owner'], 'acc-tqs4c')
    self.assertTrue(extra['public'])
    self.assertEqual(extra['source'], 'oneiric-i386-20178.gz')
    self.assertEqual(extra['source_type'], 'upload')
    self.assertEqual(extra['status'], 'deprecated')
    self.assertEqual(extra['username'], 'ubuntu')
    self.assertEqual(extra['virtual_size'], 1025)
    self.assertFalse('ancestor' in extra)
    self.assertFalse('licence_name' in extra)