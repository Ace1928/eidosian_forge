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
def test_ex_list_security_groups_with_ex_filters(self):
    ECSMockHttp.type = 'list_sgs_filters'
    self.vpc_id = 'vpc1'
    ex_filters = {'VpcId': self.vpc_id}
    sgs = self.driver.ex_list_security_groups(ex_filters=ex_filters)
    self.assertEqual(1, len(sgs))