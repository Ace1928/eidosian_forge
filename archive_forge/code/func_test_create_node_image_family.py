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
def test_create_node_image_family(self):
    node_name = 'node-name'
    size = self.driver.ex_get_size('n1-standard-1')
    node = self.driver.create_node(node_name, size, image=None, ex_image_family='coreos-stable')
    self.assertTrue(isinstance(node, Node))
    self.assertEqual(node.name, node_name)
    image = self.driver.ex_get_image('debian-7')
    self.assertRaises(ValueError, self.driver.create_node, node_name, size, image, ex_image_family='coreos-stable')