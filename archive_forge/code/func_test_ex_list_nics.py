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
def test_ex_list_nics(self):
    _, fixture = self.driver.connection.connection._load_fixture('listNics_default.json')
    fixture_nic = fixture['listnicsresponse']['nic']
    vm = self.driver.list_nodes()[0]
    nics = self.driver.ex_list_nics(vm)
    for i, nic in enumerate(nics):
        self.assertEqual(nic.id, fixture_nic[i]['id'])
        self.assertEqual(nic.network_id, fixture_nic[i]['networkid'])
        self.assertEqual(nic.net_mask, fixture_nic[i]['netmask'])
        self.assertEqual(nic.gateway, fixture_nic[i]['gateway'])
        self.assertEqual(nic.ip_address, fixture_nic[i]['ipaddress'])
        self.assertEqual(nic.is_default, fixture_nic[i]['isdefault'])
        self.assertEqual(nic.mac_address, fixture_nic[i]['macaddress'])