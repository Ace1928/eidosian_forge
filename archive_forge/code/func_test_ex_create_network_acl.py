import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_ex_create_network_acl(self):
    _, fixture = self.driver.connection.connection._load_fixture('createNetworkACL_default.json')
    fixture_network_acllist = fixture['createnetworkaclresponse']
    acllist = self.driver.ex_list_network_acllists()[0]
    network_acl = self.driver.ex_create_network_acl(protocol='test_acllist', acl_id=acllist.id, cidr_list='', start_port='80', end_port='80')
    self.assertEqual(network_acl.id, fixture_network_acllist['id'])