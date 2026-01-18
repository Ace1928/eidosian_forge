import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ROUTE53
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.route53 import Route53DNSDriver
def test_create_multi_value_record(self):
    zone = self.driver.list_zones()[0]
    records = self.driver.ex_create_multi_value_record(name='balancer', zone=zone, type=RecordType.A, data='127.0.0.1\n127.0.0.2', extra={'ttl': 0})
    self.assertEqual(len(records), 2)
    self.assertEqual(records[0].id, 'A:balancer')
    self.assertEqual(records[1].id, 'A:balancer')
    self.assertEqual(records[0].name, 'balancer')
    self.assertEqual(records[1].name, 'balancer')
    self.assertEqual(records[0].zone, zone)
    self.assertEqual(records[1].zone, zone)
    self.assertEqual(records[0].type, RecordType.A)
    self.assertEqual(records[1].type, RecordType.A)
    self.assertEqual(records[0].data, '127.0.0.1')
    self.assertEqual(records[1].data, '127.0.0.2')