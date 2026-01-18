import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_security_groups_nova(self):
    self.register_uris([dict(method='GET', uri='{endpoint}/os-security-groups?project_id=42'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_groups': []})])
    self.cloud.secgroup_source = 'nova'
    self.has_neutron = False
    self.cloud.list_security_groups(filters={'project_id': 42})
    self.assert_calls()