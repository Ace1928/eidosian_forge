from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import _proxy
from openstack.baremetal_introspection.v1 import introspection
from openstack.baremetal_introspection.v1 import introspection_rule
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
def test_introspection_rule_delete_ignore(self):
    self.verify_delete(self.proxy.delete_introspection_rule, introspection_rule.IntrospectionRule, True)