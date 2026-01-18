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
def test_ex_create_vpc(self):
    _, fixture = self.driver.connection.connection._load_fixture('createVPC_default.json')
    fixture_vpc = fixture['createvpcresponse']
    vpcoffer = self.driver.ex_list_vpc_offerings()[0]
    vpc = self.driver.ex_create_vpc(cidr='10.1.1.0/16', display_text='cloud.local', name='cloud.local', vpc_offering=vpcoffer, zone_id='2')
    self.assertEqual(vpc.id, fixture_vpc['id'])