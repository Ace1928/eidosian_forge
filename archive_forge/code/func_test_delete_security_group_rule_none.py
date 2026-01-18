import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_security_group_rule_none(self):
    self.has_neutron = False
    self.cloud.secgroup_source = None
    self.assertRaises(openstack.cloud.OpenStackCloudUnavailableFeature, self.cloud.delete_security_group_rule, '')