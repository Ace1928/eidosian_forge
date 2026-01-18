from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def check_update_complete(self, prop_diff):
    if prop_diff:
        return self.client_plugin().check_ext_resource_status('tap_flow', self.resource_id)
    return True