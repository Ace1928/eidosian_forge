from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_address_scope_handle_delete_resource_id_is_none(self):
    self.my_address_scope.resource_id = None
    self.assertIsNone(self.my_address_scope.handle_delete())
    self.assertEqual(0, self.neutronclient.delete_address_scope.call_count)