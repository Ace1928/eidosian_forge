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
def test_get_user_without_enabled(self):
    user = self.auth_instance.get_user(user_id='c')
    self.assertEqual(user.id, 'c')
    self.assertEqual(user.name, 'userwithoutenabled')
    self.assertIsNone(user.enabled)