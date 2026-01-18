import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_get_zone_zone_not_exists(self):
    PointDNSMockHttp.type = 'GET_ZONE_NOT_EXIST'
    try:
        self.driver.get_zone(zone_id='1')
    except ZoneDoesNotExistError:
        pass
    else:
        self.fail('Exception was not thrown')