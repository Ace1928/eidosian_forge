import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_server_security_groups_bad_source(self):
    self.has_neutron = False
    self.cloud.secgroup_source = 'invalid'
    server = dict(id='server_id')
    ret = self.cloud.list_server_security_groups(server)
    self.assertEqual([], ret)