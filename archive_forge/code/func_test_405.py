from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_405(self):
    self.assertRoute(self.m, '/fake_tenant/validate', 'GET', 'reject', 'DefaultMethodController', {'tenant_id': 'fake_tenant', 'allowed_methods': 'POST'})
    self.assertRoute(self.m, '/fake_tenant/stacks', 'PUT', 'reject', 'DefaultMethodController', {'tenant_id': 'fake_tenant', 'allowed_methods': 'GET,POST'})
    self.assertRoute(self.m, '/fake_tenant/stacks/fake_stack/stack_id', 'POST', 'reject', 'DefaultMethodController', {'tenant_id': 'fake_tenant', 'stack_name': 'fake_stack', 'stack_id': 'stack_id', 'allowed_methods': 'GET,PUT,PATCH,DELETE'})