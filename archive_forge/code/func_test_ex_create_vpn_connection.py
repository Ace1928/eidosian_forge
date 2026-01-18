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
def test_ex_create_vpn_connection(self):
    vpn_customer_gateway = self.driver.ex_list_vpn_customer_gateways()[0]
    vpn_gateway = self.driver.ex_list_vpn_gateways()[0]
    vpn_connection = self.driver.ex_create_vpn_connection(vpn_customer_gateway, vpn_gateway)
    self.assertEqual(vpn_connection.id, 'f45c3af8-f909-4f16-9d40-ed4409c575f8')
    self.assertEqual(vpn_connection.passive, False)
    self.assertEqual(vpn_connection.vpn_customer_gateway_id, 'ea67eaae-1c2a-4e65-b910-441e77f69bea')
    self.assertEqual(vpn_connection.vpn_gateway_id, 'cffa0cab-d1da-42a7-92f6-41379267a29f')
    self.assertEqual(vpn_connection.state, 'Connected')