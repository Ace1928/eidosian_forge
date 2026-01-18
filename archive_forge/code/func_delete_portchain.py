from unittest import mock
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron.sfc import port_chain
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def delete_portchain(self):
    mock_pc_delete = self.test_client_plugin.delete_ext_resource
    self.test_resource.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    mock_pc_delete.return_value = None
    self.assertIsNone(self.test_resource.handle_delete())
    mock_pc_delete.assert_called_once_with('port_chain', self.test_resource.resource_id)