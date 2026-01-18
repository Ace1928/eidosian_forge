from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import server_remote_console
from openstack.tests.unit import base
def test_create_type_mks_old(self):
    sot = server_remote_console.ServerRemoteConsole(server_id='fake_server', type='webmks')

    class FakeEndpointData:
        min_microversion = '2'
        max_microversion = '2.5'
    self.sess.get_endpoint_data.return_value = FakeEndpointData()
    self.assertRaises(ValueError, sot.create, self.sess)