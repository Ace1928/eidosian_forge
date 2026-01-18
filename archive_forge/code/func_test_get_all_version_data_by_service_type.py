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
def test_get_all_version_data_by_service_type(self):
    nova_disc = fixture.DiscoveryList(v2=False, v3=False)
    nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2')
    nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2.1', min_version='2.1', max_version='2.35')
    self.stub_url('GET', [], base_url=self.TEST_COMPUTE_PUBLIC, json=nova_disc)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    data = s.get_all_version_data(interface='public', service_type='compute')
    self.assertEqual({'RegionOne': {'public': {'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.1'}]}}}, data)