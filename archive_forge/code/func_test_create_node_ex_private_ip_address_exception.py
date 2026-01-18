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
def test_create_node_ex_private_ip_address_exception(self):
    name = 'test_create_node_ex_private_ip_address_exception'
    self.assertRaises(AttributeError, self.driver.create_node, name=name, image=self.fake_image, size=self.fake_size, ex_security_group_id='sg-id1', ex_private_ip_address='1.1.1.2')