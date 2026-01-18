import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GODADDY
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.godaddy import GoDaddyDNSDriver
def test_ex_check_availability(self):
    check = self.driver.ex_check_availability('wazzlewobbleflooble.com')
    self.assertEqual(check.available, True)
    self.assertEqual(check.price, 14.99)