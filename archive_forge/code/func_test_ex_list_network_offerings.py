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
def test_ex_list_network_offerings(self):
    _, fixture = self.driver.connection.connection._load_fixture('listNetworkOfferings_default.json')
    fixture_networkoffers = fixture['listnetworkofferingsresponse']['networkoffering']
    networkoffers = self.driver.ex_list_network_offerings()
    for i, networkoffer in enumerate(networkoffers):
        self.assertEqual(networkoffer.id, fixture_networkoffers[i]['id'])
        self.assertEqual(networkoffer.name, fixture_networkoffers[i]['name'])
        self.assertEqual(networkoffer.display_text, fixture_networkoffers[i]['displaytext'])
        self.assertEqual(networkoffer.for_vpc, fixture_networkoffers[i]['forvpc'])
        self.assertEqual(networkoffer.guest_ip_type, fixture_networkoffers[i]['guestiptype'])
        self.assertEqual(networkoffer.service_offering_id, fixture_networkoffers[i]['serviceofferingid'])