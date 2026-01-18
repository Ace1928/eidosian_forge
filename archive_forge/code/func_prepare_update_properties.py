from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
def prepare_update_properties(self, prop_diff):
    """Prepares prop_diff values for correct neutron update call.

        1. Merges value_specs
        2. Defaults resource name to physical resource name if None
        """
    if 'value_specs' in prop_diff:
        NeutronResource.merge_value_specs(prop_diff, self.properties[self.VALUE_SPECS])
    if 'name' in prop_diff and prop_diff['name'] is None:
        prop_diff['name'] = self.physical_resource_name()