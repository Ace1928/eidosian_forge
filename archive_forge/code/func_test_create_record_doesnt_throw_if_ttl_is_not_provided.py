import sys
import json
import unittest
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_GANDI_LIVE
from libcloud.common.gandi_live import JsonParseError, GandiLiveBaseError, InvalidRequestError
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.gandi_live import GandiLiveDNSDriver
from libcloud.test.common.test_gandi_live import BaseGandiLiveMockHttp
def test_create_record_doesnt_throw_if_ttl_is_not_provided(self):
    record = self.driver.create_record('alice', self.test_zone, 'AAAA', '::1')
    self.assertEqual(record.id, 'AAAA:alice')
    self.assertEqual(record.name, 'alice')
    self.assertEqual(record.type, RecordType.AAAA)
    self.assertEqual(record.data, '::1')