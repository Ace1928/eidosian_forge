import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_get_record_success(self):
    LiquidWebMockHttp.type = 'GET_RECORD_SUCCESS'
    record = self.driver.get_record(zone_id='13', record_id='13')
    self.assertEqual(record.id, '13')
    self.assertEqual(record.type, 'A')
    self.assertEqual(record.name, 'nerd.domain.com')
    self.assertEqual(record.data, '127.0.0.1')