import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_WORLDWIDEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.common.worldwidedns import InvalidDomainName, NonExistentDomain
from libcloud.dns.drivers.worldwidedns import WorldWideDNSError, WorldWideDNSDriver
def test_create_record_max_entry_reached(self):
    zone = self.driver.list_zones()[0]
    WorldWideDNSMockHttp.type = 'CREATE_RECORD_MAX_ENTRIES'
    record = self.driver.create_record(name='domain40', zone=zone, type=RecordType.A, data='0.0.0.40')
    WorldWideDNSMockHttp.type = 'CREATE_RECORD'
    zone = record.zone
    try:
        self.driver.create_record(name='domain41', zone=zone, type=RecordType.A, data='0.0.0.41')
    except WorldWideDNSError as e:
        self.assertEqual(e.value, 'All record entries are full')
    else:
        self.fail('Exception was not thrown')