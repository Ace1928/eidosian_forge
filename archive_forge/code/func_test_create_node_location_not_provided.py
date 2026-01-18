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
def test_create_node_location_not_provided(self):
    node_name = 'node-name'
    size = self.driver.ex_get_size('n1-standard-1')
    del size.extra['zone']
    image = self.driver.ex_get_image('debian-7')
    self.driver.zone = None
    expected_msg = 'Zone not provided'
    self.assertRaisesRegex(ValueError, expected_msg, self.driver.create_node, node_name, size, image)