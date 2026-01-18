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
def test_get_instance_vhd(self):
    with mock.patch.object(self.driver, '_ex_delete_old_vhd'):
        vhd_url = self.driver._get_instance_vhd(name='test1', ex_resource_group='000000', ex_storage_account='sga1')
        self.assertEqual(vhd_url, 'https://sga1.blob.core.windows.net/vhds/test1-os_0.vhd')
        self.driver.connection.storage_suffix = '.core.chinacloudapi.cn'
        vhd_url = self.driver._get_instance_vhd(name='test1', ex_resource_group='000000', ex_storage_account='sga1')
        self.assertEqual(vhd_url, 'https://sga1.blob.core.chinacloudapi.cn/vhds/test1-os_0.vhd')