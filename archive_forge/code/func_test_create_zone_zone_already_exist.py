import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DURABLEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.durabledns import (
def test_create_zone_zone_already_exist(self):
    DurableDNSMockHttp.type = 'ZONE_ALREADY_EXIST'
    try:
        self.driver.create_zone(domain='myzone.com.')
    except ZoneAlreadyExistsError:
        pass
    else:
        self.fail('Exception was not thrown')