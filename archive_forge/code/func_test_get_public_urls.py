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
def test_get_public_urls(self):
    data = self.fixtures.load('_v2_0__auth.json')
    data = json.loads(data)
    service_catalog = data['access']['serviceCatalog']
    catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='2.0')
    public_urls = catalog.get_public_urls(service_type='object-store')
    expected_urls = ['https://storage101.lon1.clouddrive.com/v1/MossoCloudFS_11111-111111111-1111111111-1111111', 'https://storage101.ord1.clouddrive.com/v1/MossoCloudFS_11111-111111111-1111111111-1111111']
    self.assertEqual(public_urls, expected_urls)