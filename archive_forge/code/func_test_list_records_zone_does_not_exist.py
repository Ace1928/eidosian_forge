import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE, DNS_KEYWORD_PARAMS_LINODE
from libcloud.common.linode import LinodeException
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV3
from libcloud.test.file_fixtures import DNSFileFixtures
def test_list_records_zone_does_not_exist(self):
    zone = self.driver.list_zones()[0]
    LinodeMockHttp.type = 'ZONE_DOES_NOT_EXIST'
    try:
        self.driver.list_records(zone=zone)
    except ZoneDoesNotExistError as e:
        self.assertEqual(e.zone_id, zone.id)
    else:
        self.fail('Exception was not thrown')