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
def test_get_all_version_data(self):
    cinder_disc = fixture.DiscoveryList(v2=False, v3=False)
    cinder_disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v3'].discovery.public, id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
    cinder_disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v2'].discovery.public, id='v2.0', status='SUPPORTED')
    self.stub_url('GET', [], base_url=self.TEST_VOLUME.unversioned.public + '/', json=cinder_disc)
    nova_disc = fixture.DiscoveryList(v2=False, v3=False)
    nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2')
    nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2.1', min_version='2.1', max_version='2.35')
    self.stub_url('GET', [], base_url=self.TEST_COMPUTE_PUBLIC, json=nova_disc)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    data = s.get_all_version_data(interface='public')
    self.assertEqual({'RegionOne': {'public': {'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/public/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/public/v3', 'version': '3.0'}], 'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.1'}]}}}, data)