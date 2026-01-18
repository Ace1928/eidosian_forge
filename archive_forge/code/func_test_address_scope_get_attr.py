from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_address_scope_get_attr(self):
    self.my_address_scope.resource_id = 'addrs_id'
    addrs = {'address_scope': {'name': 'test_addrs', 'id': '9c1eb3fe-7bba-479d-bd43-1d497e53c384', 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0', 'shared': True, 'ip_version': 4}}
    self.neutronclient.show_address_scope.return_value = addrs
    self.assertEqual(addrs['address_scope'], self.my_address_scope.FnGetAtt('show'))
    self.neutronclient.show_address_scope.assert_called_once_with(self.my_address_scope.resource_id)