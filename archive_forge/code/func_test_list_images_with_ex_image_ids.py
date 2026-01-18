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
def test_list_images_with_ex_image_ids(self):
    ECSMockHttp.type = 'list_images_ex_image_ids'
    self.driver.list_images(location=self.fake_location, ex_image_ids=[self.fake_image.id, 'not-existed'])