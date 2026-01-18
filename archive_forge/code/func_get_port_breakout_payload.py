from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_port_breakout_payload(self, name, mode, match):
    payload = {}
    speed = get_speed_from_breakout_mode(mode)
    if speed:
        num_breakouts = int(mode[0])
        mode_cfg = {'groups': {'group': [{'index': 1, 'config': {'index': 1, 'num-breakouts': num_breakouts, 'breakout-speed': speed}}]}}
        port_cfg = {'openconfig-platform-port:breakout-mode': mode_cfg}
        compo_cfg = {'name': name, 'port': port_cfg}
        payload = {'openconfig-platform:components': {'component': [compo_cfg]}}
    return payload