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
def test_create_node_with_metadata(self):
    node_name = 'node-name'
    image = self.driver.ex_get_image('debian-7')
    size = self.driver.ex_get_size('n1-standard-1')
    zone = self.driver.ex_get_zone('us-central1-a')
    md = [{'key': 'k0', 'value': 'v0'}, {'key': 'k1', 'value': 'v1'}]
    request, data = self.driver._create_node_req(node_name, size, image, zone, metadata=md)
    self.assertTrue('items' in data['metadata'])
    self.assertEqual(len(data['metadata']['items']), 2)
    md = {'key': 'key1', 'value': 'value1'}
    request, data = self.driver._create_node_req(node_name, size, image, zone, metadata=md)
    self.assertTrue('items' in data['metadata'])
    self.assertEqual(len(data['metadata']['items']), 1)
    md = {'items': [{'key': 'k0', 'value': 'v0'}]}
    request, data = self.driver._create_node_req(node_name, size, image, zone, metadata=md)
    self.assertTrue('items' in data['metadata'])
    self.assertEqual(len(data['metadata']['items']), 1)
    self.assertEqual(data['metadata']['items'][0]['key'], 'k0')
    self.assertEqual(data['metadata']['items'][0]['value'], 'v0')