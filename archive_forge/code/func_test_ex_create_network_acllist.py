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
def test_ex_create_network_acllist(self):
    _, fixture = self.driver.connection.connection._load_fixture('createNetworkACLList_default.json')
    fixture_network_acllist = fixture['createnetworkacllistresponse']
    vpc = self.driver.ex_list_vpcs()[0]
    network_acllist = self.driver.ex_create_network_acllist(name='test_acllist', vpc_id=vpc.id, description='test description')
    self.assertEqual(network_acllist.id, fixture_network_acllist['id'])