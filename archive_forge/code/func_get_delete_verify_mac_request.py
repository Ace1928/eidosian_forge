from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_verify_mac_request(self, afi):
    """makes and returns request to "delete" aka reset to default the config for one afi family's verify mac setting"""
    payload = {'openconfig-dhcp-snooping:dhcpv{v}-verify-mac-address'.format(v=self.afi_to_vnum(afi)): True}
    return [{'path': self.verify_mac_uri.format(v=self.afi_to_vnum(afi)), 'method': self.patch_method_value, 'data': payload}]