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
def test_get_zone_non_exist(self):
    try:
        self.driver.get_zone('nonexists.example.com')
        self.fail('expected a ZoneDoesNotExistError')
    except ZoneDoesNotExistError:
        pass
    except Exception:
        raise