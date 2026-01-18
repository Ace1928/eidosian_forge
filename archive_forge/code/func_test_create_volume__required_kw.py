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
def test_create_volume__required_kw(self):
    location = self.driver.list_locations()[0]
    fn = functools.partial(self.driver.create_volume, 2, 'test-disk-1')
    self.assertRaises(ValueError, fn)
    self.assertRaises(ValueError, fn, location=location)
    self.assertRaises(ValueError, fn, ex_resource_group='000000')
    ret_value = fn(ex_resource_group='000000', location=location)
    self.assertTrue(isinstance(ret_value, StorageVolume))