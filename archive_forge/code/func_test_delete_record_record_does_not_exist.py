import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nsone import NsOneException
from libcloud.test.secrets import DNS_PARAMS_NSONE
from libcloud.dns.drivers.nsone import NsOneDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_delete_record_record_does_not_exist(self):
    NsOneMockHttp.type = 'DELETE_RECORD_RECORD_DOES_NOT_EXIST'
    try:
        self.driver.delete_record(record=self.test_record)
    except RecordDoesNotExistError as e:
        self.assertEqual(e.record_id, self.test_record.id)
    else:
        self.fail('Exception was not thrown')