import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ZONOMI
from libcloud.dns.drivers.zonomi import ZonomiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_get_record_does_not_exist(self):
    ZonomiMockHttp.type = 'GET_RECORD_DOES_NOT_EXIST'
    zone = Zone(id='zone.com', domain='zone.com', type='master', ttl=None, driver=self.driver)
    self.driver.get_zone = MagicMock(return_value=zone)
    record_id = 'nonexistent'
    try:
        self.driver.get_record(record_id=record_id, zone_id='zone.com')
    except RecordDoesNotExistError as e:
        self.assertEqual(e.record_id, record_id)
    else:
        self.fail('Exception was not thrown.')