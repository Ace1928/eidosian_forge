import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_ex_get_disktype(self):
    disktype_name = 'pd-ssd'
    disktype_zone = 'us-central1-a'
    disktype = self.driver.ex_get_disktype(disktype_name, disktype_zone)
    self.assertEqual(disktype.name, disktype_name)
    self.assertEqual(disktype.zone.name, disktype_zone)
    self.assertEqual(disktype.extra['description'], 'SSD Persistent Disk')
    self.assertEqual(disktype.extra['valid_disk_size'], '10GB-10240GB')
    self.assertEqual(disktype.extra['default_disk_size_gb'], '100')
    disktype_name = 'pd-ssd'
    disktype = self.driver.ex_get_disktype(disktype_name)
    self.assertEqual(disktype.name, disktype_name)