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
def test_list_locations(self):
    locations = self.driver.list_locations()
    self.assertEqual(9, len(locations))
    location = locations[0]
    self.assertEqual('ap-southeast-1', location.id)
    self.assertIsNone(location.country)