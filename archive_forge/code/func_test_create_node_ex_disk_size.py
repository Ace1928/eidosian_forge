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
def test_create_node_ex_disk_size(self):
    location = NodeLocation('any_location', '', '', self.driver)
    size = NodeSize('any_size', '', 0, 0, 0, 0, driver=self.driver)
    image = AzureImage('1', '1', 'ubuntu', 'pub', location.id, self.driver)
    auth = NodeAuthPassword('any_password')
    node = self.driver.create_node('test-node-1', size, image, auth, location=location, ex_resource_group='000000', ex_storage_account='000000', ex_user_name='any_user', ex_network='000000', ex_subnet='000000', ex_disk_size=100, ex_use_managed_disks=True)
    hardware_profile = node.extra['properties']['hardwareProfile']
    os_profile = node.extra['properties']['osProfile']
    storage_profile = node.extra['properties']['storageProfile']
    self.assertEqual(node.name, 'test-node-1')
    self.assertEqual(node.state, NodeState.UPDATING)
    self.assertEqual(node.private_ips, ['10.0.0.1'])
    self.assertEqual(node.public_ips, [])
    self.assertEqual(node.extra['location'], location.id)
    self.assertEqual(hardware_profile['vmSize'], size.id)
    self.assertEqual(os_profile['adminUsername'], 'any_user')
    self.assertEqual(os_profile['adminPassword'], 'any_password')
    self.assertTrue('managedDisk' in storage_profile['osDisk'])
    self.assertEqual(storage_profile['osDisk']['diskSizeGB'], 100)
    self.assertTrue('deleteOption' not in storage_profile['osDisk'])
    self.assertTrue(storage_profile['imageReference'], {'publisher': image.publisher, 'offer': image.offer, 'sku': image.sku, 'version': image.version})