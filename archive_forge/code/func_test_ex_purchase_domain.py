import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GODADDY
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.godaddy import GoDaddyDNSDriver
def test_ex_purchase_domain(self):
    fixtures = DNSFileFixtures('godaddy')
    document = fixtures.load('purchase_request.json')
    order = self.driver.ex_purchase_domain(document)
    self.assertEqual(order.order_id, 1)