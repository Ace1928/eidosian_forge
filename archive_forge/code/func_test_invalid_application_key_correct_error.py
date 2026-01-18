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
def test_invalid_application_key_correct_error(self):
    OvhMockHttp.type = 'invalid_app_key_error'
    driver = OvhNodeDriver('appkeyinvalid', 'application_secret', 'project_id', 'consumer_key')
    expected_msg = 'Invalid application key'
    self.assertRaisesRegex(BaseHTTPError, expected_msg, driver.list_nodes)