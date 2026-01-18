import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_security_groups_none(self):
    self.cloud.secgroup_source = None
    self.has_neutron = False
    self.assertRaises(openstack.cloud.OpenStackCloudUnavailableFeature, self.cloud.list_security_groups)