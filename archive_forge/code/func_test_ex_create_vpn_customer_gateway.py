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
def test_ex_create_vpn_customer_gateway(self):
    vpn_customer_gateway = self.driver.ex_create_vpn_customer_gateway(cidr_list='10.0.0.0/24', esp_policy='3des-md5', gateway='10.0.0.1', ike_policy='3des-md5', ipsec_psk='ipsecpsk')
    self.assertEqual(vpn_customer_gateway.id, 'cef3c766-116a-4e83-9844-7d08ab7d3fd4')
    self.assertEqual(vpn_customer_gateway.esp_policy, '3des-md5')
    self.assertEqual(vpn_customer_gateway.gateway, '10.0.0.1')
    self.assertEqual(vpn_customer_gateway.ike_policy, '3des-md5')
    self.assertEqual(vpn_customer_gateway.ipsec_psk, 'ipsecpsk')