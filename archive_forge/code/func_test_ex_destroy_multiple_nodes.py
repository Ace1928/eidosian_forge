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
def test_ex_destroy_multiple_nodes(self):
    nodes = []
    nodes.append(self.driver.ex_get_node('lcnode-000'))
    nodes.append(self.driver.ex_get_node('lcnode-001'))
    destroyed = self.driver.ex_destroy_multiple_nodes(nodes)
    for d in destroyed:
        self.assertTrue(d)