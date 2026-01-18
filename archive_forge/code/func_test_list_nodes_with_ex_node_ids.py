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
def test_list_nodes_with_ex_node_ids(self):
    ECSMockHttp.type = 'list_nodes_ex_node_ids'
    nodes = self.driver.list_nodes(ex_node_ids=['i-28n7dkvov', 'not-existed-id'])
    self.assertIsNotNone(nodes)