import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
def test_list_supported_versions(self):
    OpenStackIdentity_3_0_MockHttp.type = 'v3'
    versions = self.auth_instance.list_supported_versions()
    self.assertEqual(len(versions), 2)
    self.assertEqual(versions[0].version, 'v2.0')
    self.assertEqual(versions[0].url, 'http://192.168.18.100:5000/v2.0/')
    self.assertEqual(versions[1].version, 'v3.0')
    self.assertEqual(versions[1].url, 'http://192.168.18.100:5000/v3/')