from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_address_scope_handle_create(self):
    addrs = {'address_scope': {'id': '9c1eb3fe-7bba-479d-bd43-1d497e53c384', 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0', 'shared': False, 'ip_version': 4}}
    create_props = {'name': 'test_address_scope', 'shared': False, 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0', 'ip_version': 4}
    self.neutronclient.create_address_scope.return_value = addrs
    self.my_address_scope.handle_create()
    self.assertEqual('9c1eb3fe-7bba-479d-bd43-1d497e53c384', self.my_address_scope.resource_id)
    self.neutronclient.create_address_scope.assert_called_once_with({'address_scope': create_props})