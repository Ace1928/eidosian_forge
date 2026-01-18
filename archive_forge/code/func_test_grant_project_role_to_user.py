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
def test_grant_project_role_to_user(self):
    project = self.auth_instance.list_projects()[0]
    role = self.auth_instance.list_roles()[0]
    user = self.auth_instance.list_users()[0]
    result = self.auth_instance.grant_project_role_to_user(project=project, role=role, user=user)
    self.assertTrue(result)