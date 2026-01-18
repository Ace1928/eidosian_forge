import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.dns.base import Zone
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError
from libcloud.test.secrets import DNS_PARAMS_AURORADNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.auroradns import AuroraDNSDriver, AuroraDNSHealthCheckType
def test_403_status_code(self):
    AuroraDNSDriverMockHttp.type = 'HTTP_FORBIDDEN'
    with self.assertRaises(ProviderError) as ctx:
        self.driver.list_zones()
    self.assertEqual(ctx.exception.value, 'Authorization failed')
    self.assertEqual(ctx.exception.http_code, 403)