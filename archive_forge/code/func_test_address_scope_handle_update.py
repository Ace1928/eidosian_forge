from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_address_scope_handle_update(self):
    addrs_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.my_address_scope.resource_id = addrs_id
    props = {'name': 'test_address_scope', 'shared': True}
    update_dict = props.copy()
    update_snippet = rsrc_defn.ResourceDefinition(self.my_address_scope.name, self.my_address_scope.type(), props)
    self.my_address_scope.handle_update(json_snippet=update_snippet, tmpl_diff={}, prop_diff=props)
    props['name'] = None
    self.my_address_scope.handle_update(json_snippet=update_snippet, tmpl_diff={}, prop_diff=props)
    self.assertEqual(2, self.neutronclient.update_address_scope.call_count)
    self.neutronclient.update_address_scope.assert_called_with(addrs_id, {'address_scope': update_dict})