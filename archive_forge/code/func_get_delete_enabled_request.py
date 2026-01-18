from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_enabled_request(self, afi):
    """makes and returns request to "delete" aka reset to default the enabled setting for one afi family. returns as a list"""
    payload = {'openconfig-dhcp-snooping:dhcpv{v}-admin-enable'.format(v=self.afi_to_vnum(afi)): False}
    return [{'path': self.enable_uri.format(v=self.afi_to_vnum(afi)), 'method': self.patch_method_value, 'data': payload}]