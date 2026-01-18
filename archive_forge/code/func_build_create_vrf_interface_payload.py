from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def build_create_vrf_interface_payload(self, conf):
    members = conf['members'].get('interfaces', None)
    network_inst_payload = dict()
    if members:
        network_inst_payload.update({'openconfig-network-instance:interface': []})
        for member in members:
            if member['name']:
                member_config_payload = dict({'id': member['name']})
                member_payload = dict({'id': member['name'], 'config': member_config_payload})
                network_inst_payload['openconfig-network-instance:interface'].append(member_payload)
    return network_inst_payload