from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_replaced_config(self, have, want):
    replaced_config = dict()
    h_hostname = have.get('hostname', None)
    w_hostname = want.get('hostname', None)
    if h_hostname != w_hostname and w_hostname:
        replaced_config = have.copy()
        return replaced_config
    h_intf_name = have.get('interface_naming', None)
    w_intf_name = want.get('interface_naming', None)
    if h_intf_name != w_intf_name and w_intf_name:
        replaced_config = have.copy()
        return replaced_config
    h_ac_addr = have.get('anycast_address', None)
    w_ac_addr = want.get('anycast_address', None)
    if h_ac_addr != w_ac_addr and w_ac_addr:
        replaced_config['anycast_address'] = h_ac_addr
        return replaced_config
    return replaced_config