import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_create_mail_redirect(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    PointDNSMockHttp.type = 'CREATE'
    mail_redirect = self.driver.ex_create_mail_redirect('user@example-site.com', 'admin', zone)
    self.assertEqual(mail_redirect.id, '5')
    self.assertEqual(mail_redirect.source, 'admin')
    self.assertEqual(mail_redirect.destination, 'user@example-site.com')
    self.assertEqual(zone.id, mail_redirect.zone.id)