import sys
import datetime
import tempfile
from unittest.mock import Mock
from libcloud import __version__
from libcloud.test import unittest
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import assertRegex
def test_export_zone_to_bind_format_slave_should_throw(self):
    zone = Zone(id=1, domain='example.com', type='slave', ttl=900, driver=self.driver)
    self.assertRaises(ValueError, zone.export_to_bind_format)