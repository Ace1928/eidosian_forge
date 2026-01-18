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
def test_ex_instancegroupmanager_set_autohealing_policies(self):
    kwargs = {'host': 'lchost', 'path': '/lc', 'port': 8000, 'interval': 10, 'timeout': 10, 'unhealthy_threshold': 4, 'healthy_threshold': 3, 'description': 'test healthcheck'}
    healthcheck_name = 'lchealthcheck'
    hc = self.driver.ex_create_healthcheck(healthcheck_name, **kwargs)
    ig_name = 'myinstancegroup'
    ig_zone = 'us-central1-a'
    manager = self.driver.ex_get_instancegroupmanager(ig_name, ig_zone)
    res = self.driver.ex_instancegroupmanager_set_autohealingpolicies(manager=manager, healthcheck=hc, initialdelaysec=2)
    self.assertTrue(res)
    res = manager.set_autohealingpolicies(healthcheck=hc, initialdelaysec=2)
    self.assertTrue(res)