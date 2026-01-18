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
def test_ex_get_instancegroupmanager(self):
    igmgr_name = 'myinstancegroup'
    igmgr = self.driver.ex_get_instancegroupmanager(igmgr_name, 'us-central1-b')
    self.assertEqual(igmgr.name, igmgr_name)
    self.assertEqual(igmgr.size, 4)
    self.assertEqual(igmgr.zone.name, 'us-central1-b')
    igmgr = self.driver.ex_get_instancegroupmanager(igmgr_name)
    self.assertEqual(igmgr.name, igmgr_name)
    self.assertEqual(igmgr.size, 4)
    self.assertEqual(igmgr.zone.name, 'us-central1-a')