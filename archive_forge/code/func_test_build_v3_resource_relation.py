import copy
from testtools import matchers
from keystone.common import json_home
from keystone.tests import unit
def test_build_v3_resource_relation(self):
    resource_name = self.getUniqueString()
    relation = json_home.build_v3_resource_relation(resource_name)
    exp_relation = 'https://docs.openstack.org/api/openstack-identity/3/rel/%s' % resource_name
    self.assertThat(relation, matchers.Equals(exp_relation))