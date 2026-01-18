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
def test_list_health_checks(self):
    zone = self.driver.get_zone('example.com')
    checks = self.driver.ex_list_healthchecks(zone)
    self.assertEqual(len(checks), 3)
    for check in checks:
        self.assertEqual(check.interval, 60)
        self.assertEqual(check.type, AuroraDNSHealthCheckType.HTTP)