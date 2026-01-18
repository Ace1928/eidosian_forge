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
def test_ex_create_additional_capabilities(self):
    add_cap = {'ultraSSDEnabled': True, 'hibernationEnabled': True}
    node = self.driver.list_nodes()[0]
    self.driver.ex_create_additional_capabilities(node, add_cap, '000000')
    self.assertTrue(node.extra['properties']['additionalCapabilities']['ultraSSDEnabled'])
    self.assertTrue(node.extra['properties']['additionalCapabilities']['hibernationEnabled'])