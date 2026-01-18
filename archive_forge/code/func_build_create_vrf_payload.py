from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def build_create_vrf_payload(self, conf):
    name = conf['name']
    netw_inst = dict({'name': name})
    netw_inst['config'] = dict({'name': name})
    netw_inst['config'].update({'enabled': True})
    netw_inst['config'].update({'type': 'L3VRF'})
    netw_inst_arr = [netw_inst]
    return dict({'openconfig-network-instance:network-instances': {'network-instance': netw_inst_arr}})