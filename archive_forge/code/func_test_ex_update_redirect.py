import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_update_redirect(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    redirect = self.driver.ex_get_redirect(zone.id, '36843229')
    PointDNSMockHttp.type = 'UPDATE'
    _redirect = self.driver.ex_update_redirect(redirect, 'http://updatedother.com', 'redirect3', '302')
    self.assertEqual(_redirect.id, '36843229')
    self.assertEqual(_redirect.name, 'redirect3.domain1.com.')
    self.assertEqual(_redirect.type, '302')
    self.assertEqual(_redirect.data, 'http://updatedother.com')
    self.assertIsNone(_redirect.iframe)
    self.assertEqual(_redirect.query, False)
    self.assertEqual(zone.id, _redirect.zone.id)