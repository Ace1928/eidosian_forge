import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_list_records_empty(self):
    LiquidWebMockHttp.type = 'EMPTY_RECORDS_LIST'
    zone = self.test_zone
    records = self.driver.list_records(zone=zone)
    self.assertEqual(records, [])