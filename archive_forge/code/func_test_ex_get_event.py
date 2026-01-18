import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import FileFixtures
from libcloud.common.digitalocean import DigitalOceanBaseDriver
def test_ex_get_event(self):
    action = self.driver.ex_get_event('12345670')
    self.assertEqual(action['id'], 12345670)
    self.assertEqual(action['status'], 'completed')
    self.assertEqual(action['type'], 'power_on')