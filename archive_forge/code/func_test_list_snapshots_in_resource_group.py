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
def test_list_snapshots_in_resource_group(self):
    snaps = self.driver.list_snapshots(ex_resource_group='111111')
    self.assertEqual(len(snaps), 2)
    self.assertEqual(snaps[0].name, 'test-snap-3')
    self.assertEqual(snaps[0].extra['name'], 'test-snap-3')
    self.assertEqual(snaps[0].state, VolumeSnapshotState.ERROR)
    self.assertEqual(snaps[0].extra['source_id'], '/subscriptions/99999999-9999-9999-9999-999999999999/resourceGroups/111111/providers/Microsoft.Compute/disks/test-disk-3')
    self.assertEqual(snaps[0].size, 2)
    self.assertTrue(isinstance(snaps[0].created, datetime))