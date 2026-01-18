from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def handle_delete(self):
    if self.resource_id is None:
        return
    with self.client_plugin().ignore_not_found:
        self.client_plugin().delete_ext_resource('tap_flow', self.resource_id)