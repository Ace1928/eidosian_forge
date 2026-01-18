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
def test_build_network_gce_struct(self):
    network = self.driver.ex_get_network('lcnetwork')
    address = self.driver.ex_get_address('lcaddress')
    internalip = self.driver.ex_get_address('testaddress')
    subnetwork_name = 'cf-972cf02e6ad49112'
    subnetwork = self.driver.ex_get_subnetwork(subnetwork_name)
    d = self.driver._build_network_gce_struct(network, subnetwork, address)
    self.assertTrue('network' in d)
    self.assertTrue('subnetwork' in d)
    self.assertTrue('kind' in d and d['kind'] == 'compute#instanceNetworkInterface')
    self.assertEqual(d['accessConfigs'][0]['natIP'], address.address)
    d = self.driver._build_network_gce_struct(network, subnetwork, address, internal_ip=internalip)
    self.assertTrue('network' in d)
    self.assertTrue('subnetwork' in d)
    self.assertTrue('kind' in d and d['kind'] == 'compute#instanceNetworkInterface')
    self.assertEqual(d['accessConfigs'][0]['natIP'], address.address)
    self.assertEqual(d['networkIP'], internalip)
    network = self.driver.ex_get_network('default')
    d = self.driver._build_network_gce_struct(network)
    self.assertTrue('network' in d)
    self.assertFalse('subnetwork' in d)
    self.assertTrue('kind' in d and d['kind'] == 'compute#instanceNetworkInterface')