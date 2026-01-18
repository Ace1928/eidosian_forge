from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
@property
def policy_id(self):
    if not self._policy_id:
        self._policy_id = self.client_plugin().get_qos_policy_id(self.properties[self.POLICY])
    return self._policy_id