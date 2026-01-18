import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_delete_record_RECORD_DOES_NOT_EXIST_ERROR(self):
    LiquidWebMockHttp.type = 'DELETE_RECORD_RECORD_DOES_NOT_EXIST'
    record = self.test_record
    try:
        self.driver.delete_record(record=record)
    except RecordDoesNotExistError as e:
        self.assertEqual(e.record_id, '13')
    else:
        self.fail('Exception was not thrown')