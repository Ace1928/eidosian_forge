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
def test_list_volumes_with_ex_filters(self):
    ECSMockHttp.type = 'list_volumes_ex_filters'
    ex_filters = {'InstanceId': self.fake_node.id}
    volumes = self.driver.list_volumes(ex_filters=ex_filters)
    self.assertIsNotNone(volumes)