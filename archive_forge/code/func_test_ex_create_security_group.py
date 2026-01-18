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
def test_ex_create_security_group(self):
    self.sg_description = 'description'
    self.client_token = 'client-token'
    sg_id = self.driver.ex_create_security_group(description=self.sg_description, client_token=self.client_token)
    self.assertEqual('sg-F876FF7BA', sg_id)