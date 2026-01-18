import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_create_zone_with_explicit_account(self):
    zone = self.driver.create_zone(domain='example2.com', extra={'account': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'})
    self.assertEqual(zone.id, '6789')
    self.assertEqual(zone.domain, 'example2.com')