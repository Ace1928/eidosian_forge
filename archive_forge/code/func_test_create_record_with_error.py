import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_create_record_with_error(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    PointDNSMockHttp.type = 'CREATE_WITH_ERROR'
    try:
        self.driver.create_record(name='site.example.com', zone=zone, type=RecordType.A, data='1.2.3.4')
    except PointDNSException:
        pass
    else:
        self.fail('Exception was not thrown')