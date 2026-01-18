import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_delete_record_record_not_exists(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    records = self.driver.list_records(zone=zone)
    self.assertEqual(len(records), 2)
    record = records[1]
    PointDNSMockHttp.type = 'DELETE_RECORD_NOT_EXIST'
    try:
        self.driver.delete_record(record=record)
    except RecordDoesNotExistError:
        pass
    else:
        self.fail('Exception was not thrown')