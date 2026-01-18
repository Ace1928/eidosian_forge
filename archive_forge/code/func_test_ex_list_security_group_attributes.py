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
def test_ex_list_security_group_attributes(self):
    self.sga_nictype = 'internet'
    sgas = self.driver.ex_list_security_group_attributes(group_id=self.fake_security_group_id, nic_type=self.sga_nictype)
    self.assertEqual(1, len(sgas))
    sga = sgas[0]
    self.assertEqual('ALL', sga.ip_protocol)
    self.assertEqual('-1/-1', sga.port_range)
    self.assertEqual('Accept', sga.policy)
    self.assertEqual('internet', sga.nic_type)