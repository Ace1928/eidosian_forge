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
def test_create_node_with_accelerator(self):
    node_name = 'node-name'
    image = self.driver.ex_get_image('debian-7')
    size = self.driver.ex_get_size('n1-standard-1')
    zone = self.driver.ex_get_zone('us-central1-a')
    request, data = self.driver._create_node_req(node_name, size, image, zone, ex_accelerator_type='nvidia-tesla-k80', ex_accelerator_count=3)
    self.assertTrue('guestAccelerators' in data)
    self.assertEqual(len(data['guestAccelerators']), 1)
    self.assertTrue('nvidia-tesla-k80' in data['guestAccelerators'][0]['acceleratorType'])
    self.assertEqual(data['guestAccelerators'][0]['acceleratorCount'], 3)