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
def test_ex_list_vpn_connections(self):
    vpn_connections = self.driver.ex_list_vpn_connections()
    self.assertEqual(len(vpn_connections), 1)
    self.assertEqual(vpn_connections[0].id, '8f482d9a-6cee-453b-9e78-b0e1338ffce9')
    self.assertEqual(vpn_connections[0].passive, False)
    self.assertEqual(vpn_connections[0].vpn_customer_gateway_id, 'ea67eaae-1c2a-4e65-b910-441e77f69bea')
    self.assertEqual(vpn_connections[0].vpn_gateway_id, 'cffa0cab-d1da-42a7-92f6-41379267a29f')
    self.assertEqual(vpn_connections[0].state, 'Connected')