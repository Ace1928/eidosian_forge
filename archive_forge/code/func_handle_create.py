from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def handle_create(self):
    props = self.prepare_properties(self.properties, self.physical_resource_name())
    props['source_port'] = props.pop(self.PORT)
    props['tap_service_id'] = props.pop(self.TAP_SERVICE)
    tap_flow = self.client_plugin().create_ext_resource('tap_flow', props)
    self.resource_id_set(tap_flow['id'])