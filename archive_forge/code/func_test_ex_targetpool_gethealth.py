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
def test_ex_targetpool_gethealth(self):
    targetpool = self.driver.ex_get_targetpool('lb-pool')
    health = targetpool.get_health('libcloud-lb-demo-www-000')
    self.assertEqual(len(health), 1)
    self.assertTrue('node' in health[0])
    self.assertTrue('health' in health[0])
    self.assertEqual(health[0]['health'], 'UNHEALTHY')