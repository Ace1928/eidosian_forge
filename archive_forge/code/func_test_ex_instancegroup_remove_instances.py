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
def test_ex_instancegroup_remove_instances(self):
    name = 'myname'
    loc = 'us-central1-a'
    gceobj = self.driver.ex_get_instancegroup(name, loc)
    node_name = self.driver.ex_get_node('node-name', loc)
    lcnode = self.driver.ex_get_node('lcnode-001', loc)
    node_list = [node_name, lcnode]
    self.assertTrue(self.driver.ex_instancegroup_remove_instances(gceobj, node_list))