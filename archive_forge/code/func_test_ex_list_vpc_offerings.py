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
def test_ex_list_vpc_offerings(self):
    _, fixture = self.driver.connection.connection._load_fixture('listVPCOfferings_default.json')
    fixture_vpcoffers = fixture['listvpcofferingsresponse']['vpcoffering']
    vpcoffers = self.driver.ex_list_vpc_offerings()
    for i, vpcoffer in enumerate(vpcoffers):
        self.assertEqual(vpcoffer.id, fixture_vpcoffers[i]['id'])
        self.assertEqual(vpcoffer.name, fixture_vpcoffers[i]['name'])
        self.assertEqual(vpcoffer.display_text, fixture_vpcoffers[i]['displaytext'])