import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ROUTE53
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.route53 import Route53DNSDriver
def test_create_SPF_record_quoted(self):
    """
        Check that SPF values already quoted are not changed
        """
    zone = self.driver.list_zones()[0]
    record = self.driver.create_record(name='', zone=zone, type=RecordType.SPF, data='"test"')
    self.assertEqual(record.id, 'SPF:')
    self.assertEqual(record.name, '')
    self.assertEqual(record.zone, zone)
    self.assertEqual(record.type, RecordType.SPF)
    self.assertEqual(record.data, '"test"')