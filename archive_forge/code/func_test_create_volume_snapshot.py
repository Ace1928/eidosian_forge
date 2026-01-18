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
def test_create_volume_snapshot(self):
    ECSMockHttp.type = 'create_volume_snapshot'
    self.snapshot_name = 'fake-snapshot1'
    self.description = 'fake-description'
    self.client_token = 'client-token'
    snapshot = self.driver.create_volume_snapshot(self.fake_volume, name=self.snapshot_name, ex_description=self.description, ex_client_token=self.client_token)
    self.assertIsNotNone(snapshot)