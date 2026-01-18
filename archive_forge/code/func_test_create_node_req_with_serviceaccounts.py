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
def test_create_node_req_with_serviceaccounts(self):
    image = self.driver.ex_get_image('debian-7')
    size = self.driver.ex_get_size('n1-standard-1')
    location = self.driver.zone
    network = self.driver.ex_get_network('default')
    ex_sa = [{'scopes': ['compute-ro', 'pubsub', 'storage-ro']}]
    node_request, node_data = self.driver._create_node_req('lcnode', size, image, location, network, ex_service_accounts=ex_sa)
    self.assertIsInstance(node_data['serviceAccounts'], list)
    self.assertIsInstance(node_data['serviceAccounts'][0], dict)
    self.assertEqual(node_data['serviceAccounts'][0]['email'], 'default')
    self.assertIsInstance(node_data['serviceAccounts'][0]['scopes'], list)
    self.assertEqual(len(node_data['serviceAccounts'][0]['scopes']), 3)
    self.assertTrue('https://www.googleapis.com/auth/devstorage.read_only' in node_data['serviceAccounts'][0]['scopes'])
    self.assertTrue('https://www.googleapis.com/auth/compute.readonly' in node_data['serviceAccounts'][0]['scopes'])