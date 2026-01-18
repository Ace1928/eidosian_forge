import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ZONOMI
from libcloud.dns.drivers.zonomi import ZonomiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_convert_to_slave(self):
    zone = self.test_zone
    result = self.driver.ex_convert_to_secondary(zone, '1.2.3.4')
    self.assertTrue(result)