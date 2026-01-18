import sys
import json
import functools
from datetime import datetime
from unittest import mock
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib, parse_qs, urlparse, urlunquote
from libcloud.common.types import LibcloudError
from libcloud.compute.base import NodeSize, NodeLocation, StorageVolume, VolumeSnapshot
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.azure_arm import (
def test_create_node_storage_account_not_provided_and_not_ex_use_managed_disks(self):
    location = NodeLocation('any_location', '', '', self.driver)
    size = NodeSize('any_size', '', 0, 0, 0, 0, driver=self.driver)
    image = AzureImage('1', '1', 'ubuntu', 'pub', location.id, self.driver)
    auth = NodeAuthPassword('any_password')
    expected_msg = 'ex_use_managed_disks is False, must provide ex_storage_account'
    self.assertRaisesRegex(ValueError, expected_msg, self.driver.create_node, 'test-node-1', size, image, auth, location=location, ex_resource_group='000000', ex_storage_account=None, ex_user_name='any_user', ex_network='000000', ex_subnet='000000', ex_use_managed_disks=False)
    node = self.driver.create_node('test-node-1', size, image, auth, location=location, ex_resource_group='000000', ex_storage_account=None, ex_user_name='any_user', ex_network='000000', ex_subnet='000000', ex_use_managed_disks=True)
    self.assertTrue(node)