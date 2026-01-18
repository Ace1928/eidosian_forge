import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
def test_get_service_names(self):
    data = self.fixtures.load('_v2_0__auth.json')
    data = json.loads(data)
    service_catalog = data['access']['serviceCatalog']
    catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='2.0')
    service_names = catalog.get_service_names()
    self.assertEqual(service_names, ['cinderv2', 'cinderv3', 'cloudFiles', 'cloudFilesCDN', 'cloudServers', 'cloudServersOpenStack', 'cloudServersPreprod', 'glance', 'neutron', 'nova'])
    service_names = catalog.get_service_names(service_type='compute')
    self.assertEqual(service_names, ['cloudServers', 'cloudServersOpenStack', 'cloudServersPreprod', 'nova'])