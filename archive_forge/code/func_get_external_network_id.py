from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def get_external_network_id(self, pool=None):
    if pool:
        neutron_plugin = self.client_plugin('neutron')
        return neutron_plugin.find_resourceid_by_name_or_id(neutron_plugin.RES_TYPE_NETWORK, pool)
    ext_filter = {'router:external': True}
    ext_nets = self.neutron().list_networks(**ext_filter)['networks']
    if len(ext_nets) != 1:
        raise exception.Error(_('Expected 1 external network, found %d') % len(ext_nets))
    external_network_id = ext_nets[0]['id']
    return external_network_id