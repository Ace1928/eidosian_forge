import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_list_records_success(self):
    LiquidWebMockHttp.type = 'LIST_RECORDS_SUCCESS'
    zone = self.test_zone
    records = self.driver.list_records(zone=zone)
    self.assertEqual(len(records), 3)
    record = records[0]
    self.assertEqual(record.id, '13')
    self.assertEqual(record.type, 'A')
    self.assertEqual(record.name, 'nerd.domain.com')
    self.assertEqual(record.data, '127.0.0.1')
    self.assertEqual(record.zone, self.test_zone)
    self.assertEqual(record.zone.id, '11')
    second_record = records[1]
    self.assertEqual(second_record.id, '11')
    self.assertEqual(second_record.type, 'A')
    self.assertEqual(second_record.name, 'thisboy.domain.com')
    self.assertEqual(second_record.data, '127.0.0.1')
    self.assertEqual(second_record.zone, self.test_zone)
    third_record = records[2]
    self.assertEqual(third_record.id, '10')
    self.assertEqual(third_record.type, 'A')
    self.assertEqual(third_record.name, 'visitor.domain.com')
    self.assertEqual(third_record.data, '127.0.0.1')
    self.assertEqual(third_record.zone, self.test_zone)