import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_delete_redirect(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    redirect = self.driver.ex_get_redirect(zone.id, '36843229')
    PointDNSMockHttp.type = 'DELETE'
    status = self.driver.ex_delete_redirect(redirect)
    self.assertTrue(status)