import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE, DNS_KEYWORD_PARAMS_LINODE
from libcloud.common.linode import LinodeException
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV3
from libcloud.test.file_fixtures import DNSFileFixtures
def test_delete_record_does_not_exist(self):
    zone = self.driver.list_zones()[0]
    record = self.driver.list_records(zone=zone)[0]
    LinodeMockHttp.type = 'RECORD_DOES_NOT_EXIST'
    try:
        self.driver.delete_record(record=record)
    except RecordDoesNotExistError as e:
        self.assertEqual(e.record_id, record.id)
    else:
        self.fail('Exception was not thrown')