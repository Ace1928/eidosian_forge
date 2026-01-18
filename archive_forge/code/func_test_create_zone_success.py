import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_create_zone_success(self):
    LiquidWebMockHttp.type = 'CREATE_ZONE_SUCCESS'
    zone = self.driver.create_zone(domain='test.com')
    self.assertEqual(zone.id, '13')
    self.assertEqual(zone.domain, 'test.com')
    self.assertEqual(zone.type, 'NATIVE')
    self.assertIsNone(zone.ttl)
    self.assertEqual(zone.driver, self.driver)