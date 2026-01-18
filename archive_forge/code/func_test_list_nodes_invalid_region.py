import sys
import unittest
from unittest.mock import patch
from libcloud.http import LibcloudConnection
from libcloud.test import no_internet
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import OVH_PARAMS
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ovh import OvhNodeDriver
from libcloud.test.common.test_ovh import BaseOvhMockHttp
@unittest.skipIf(no_internet(), 'Internet is not reachable')
def test_list_nodes_invalid_region(self):
    OvhNodeDriver.connectionCls.conn_class = LibcloudConnection
    driver = OvhNodeDriver(*OVH_PARAMS, region='invalid')
    expected_msg = 'invalid region argument was passed.*Used host: invalid.api.ovh.com.*'
    self.assertRaisesRegex(ValueError, expected_msg, driver.list_nodes)
    expected_msg = 'invalid region argument was passed.*Used host: invalid.api.ovh.com.*'
    self.assertRaisesRegex(ValueError, expected_msg, driver.connection.request_consumer_key, '1')