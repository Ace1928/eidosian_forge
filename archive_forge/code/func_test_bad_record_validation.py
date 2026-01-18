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
def test_bad_record_validation(self):
    with self.assertRaises(RecordError) as ctx:
        self.driver.create_record('alice', self.test_zone, 'AAAA', '1' * 1025, extra={'ttl': 400})
    self.assertTrue('Record data must be' in str(ctx.exception))
    with self.assertRaises(RecordError) as ctx:
        self.driver.create_record('alice', self.test_zone, 'AAAA', '::1', extra={'ttl': 10})
    self.assertTrue('TTL must be at least' in str(ctx.exception))
    with self.assertRaises(RecordError) as ctx:
        self.driver.create_record('alice', self.test_zone, 'AAAA', '::1', extra={'ttl': 31 * 24 * 60 * 60})
    self.assertTrue('TTL must not exceed' in str(ctx.exception))