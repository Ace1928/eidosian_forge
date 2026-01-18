from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def prefix_get_delete_single_prefix_cfg(self, prefix, cfg_prefix_set, command):
    """Create the REST API request to delete the prefix specified by the "prefix"
        input parameter from the configured prefix set specified by "cfg_prefix_set".
        Return an empty request if the prefix is not present in the confgured prefix set."""
    pfx_delete_cfg_request = {}
    if not self.prefix_in_prefix_list_cfg(prefix, cfg_prefix_set):
        return pfx_delete_cfg_request
    conf_afi = command.get('afi', None)
    if not conf_afi:
        return pfx_delete_cfg_request
    pfx_set_name = command.get('name', None)
    pfx_seq = prefix.get('sequence', None)
    pfx_val = prefix.get('prefix', None)
    pfx_ge = prefix.get('ge', None)
    pfx_le = prefix.get('le', None)
    if not pfx_seq or not pfx_val:
        return pfx_delete_cfg_request
    prefix_net = self.set_ipaddress_net_attrs(pfx_val, conf_afi)
    masklength_range_str = self.get_masklength_range_string(pfx_ge, pfx_le, prefix_net)
    prefix_string = pfx_val.replace('/', '%2F')
    extended_pfx_cfg_str = self.prefix_set_delete_prefix_uri.format(pfx_set_name, int(pfx_seq), prefix_string, masklength_range_str)
    pfx_delete_cfg_request = {'path': extended_pfx_cfg_str, 'method': DELETE}
    return pfx_delete_cfg_request