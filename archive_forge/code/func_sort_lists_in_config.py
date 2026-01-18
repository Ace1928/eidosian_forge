from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def sort_lists_in_config(self, config):
    if 'profiles' in config and config['profiles'] is not None:
        config['profiles'].sort(key=self.get_profile_name)
    if 'single_hops' in config and config['single_hops'] is not None:
        config['single_hops'].sort(key=lambda x: (x['remote_address'], x['interface'], x['vrf'], x['local_address']))
    if 'multi_hops' in config and config['multi_hops'] is not None:
        config['multi_hops'].sort(key=lambda x: (x['remote_address'], x['vrf'], x['local_address']))