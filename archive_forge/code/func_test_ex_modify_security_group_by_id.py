import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.test.secrets import ECS_PARAMS
from libcloud.compute.types import NodeState, StorageVolumeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ecs import ECSDriver
def test_ex_modify_security_group_by_id(self):
    self.sg_name = 'name'
    self.sg_description = 'description'
    result = self.driver.ex_modify_security_group_by_id(group_id=self.fake_security_group_id, name=self.sg_name, description=self.sg_description)
    self.assertTrue(result)