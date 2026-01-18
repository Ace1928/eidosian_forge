import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_get_all_version_data_all_interfaces(self):
    for interface in ('public', 'internal', 'admin'):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=getattr(self.TEST_VOLUME.versions['v3'].discovery, interface), id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
        disc.add_nova_microversion(href=getattr(self.TEST_VOLUME.versions['v2'].discovery, interface), id='v2.0', status='SUPPORTED')
        self.stub_url('GET', [], base_url=getattr(self.TEST_VOLUME.unversioned, interface) + '/', json=disc)
    for url in (self.TEST_COMPUTE_PUBLIC, self.TEST_COMPUTE_INTERNAL, self.TEST_COMPUTE_ADMIN):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_microversion(href=url, id='v2')
        disc.add_microversion(href=url, id='v2.1', min_version='2.1', max_version='2.35')
        self.stub_url('GET', [], base_url=url, json=disc)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    identity_endpoint = 'http://127.0.0.1:35357/{}/'.format(self.version)
    data = s.get_all_version_data(interface=None)
    self.assertEqual({'RegionOne': {'admin': {'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/admin/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/admin/v3', 'version': '3.0'}], 'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/admin', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/admin', 'version': '2.1'}], 'identity': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': None, 'status': 'CURRENT', 'url': identity_endpoint, 'version': self.discovery_version}]}, 'internal': {'baremetal': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': None, 'status': 'CURRENT', 'url': 'https://baremetal.example.com/internal/', 'version': None}], 'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/internal/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/internal/v3', 'version': '3.0'}], 'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/internal', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/internal', 'version': '2.1'}]}, 'public': {'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/public/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/public/v3', 'version': '3.0'}], 'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.1'}]}}}, data)