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
def test_ex_list_routers(self):
    _, fixture = self.driver.connection.connection._load_fixture('listRouters_default.json')
    fixture_routers = fixture['listroutersresponse']['router']
    routers = self.driver.ex_list_routers()
    for i, router in enumerate(routers):
        self.assertEqual(router.id, fixture_routers[i]['id'])
        self.assertEqual(router.name, fixture_routers[i]['name'])
        self.assertEqual(router.state, fixture_routers[i]['state'])
        self.assertEqual(router.public_ip, fixture_routers[i]['publicip'])
        self.assertEqual(router.vpc_id, fixture_routers[i]['vpcid'])