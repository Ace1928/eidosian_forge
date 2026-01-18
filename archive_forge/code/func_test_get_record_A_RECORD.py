import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV4
from libcloud.test.file_fixtures import DNSFileFixtures
def test_get_record_A_RECORD(self):
    LinodeMockHttpV4.type = 'A_RECORD'
    record = self.driver.get_record('123', '123')
    self.assertEqual(record.id, '123')
    self.assertEqual(record.name, 'test.example.com')
    self.assertEqual(record.type, 'A')