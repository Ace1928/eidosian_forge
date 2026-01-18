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
def test_ex_list_vpcs(self):
    _, fixture = self.driver.connection.connection._load_fixture('listVPCs_default.json')
    fixture_vpcs = fixture['listvpcsresponse']['vpc']
    vpcs = self.driver.ex_list_vpcs()
    for i, vpc in enumerate(vpcs):
        self.assertEqual(vpc.id, fixture_vpcs[i]['id'])
        self.assertEqual(vpc.display_text, fixture_vpcs[i]['displaytext'])
        self.assertEqual(vpc.name, fixture_vpcs[i]['name'])
        self.assertEqual(vpc.vpc_offering_id, fixture_vpcs[i]['vpcofferingid'])
        self.assertEqual(vpc.zone_id, fixture_vpcs[i]['zoneid'])