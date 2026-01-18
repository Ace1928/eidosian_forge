from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_floating_ips_with_filters(self):
    self.assertRaisesRegex(ValueError, "Nova-network don't support server-side", self.cloud.list_floating_ips, filters={'Foo': 42})