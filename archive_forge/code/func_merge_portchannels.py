from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.lag_interfaces.lag_interfaces import Lag_interfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def merge_portchannels(self, configs, conf):
    if len(configs) == 0:
        configs.append(conf)
    else:
        new_interface = None
        if conf.get('members') and conf['members'].get('interfaces'):
            new_interface = conf['members']['interfaces'][0]
        else:
            configs.append(conf)
        if new_interface:
            matched = next((cfg for cfg in configs if cfg['name'] == conf['name']), None)
            if matched and matched.get('members'):
                ext_interfaces = matched.get('members').get('interfaces', [])
                ext_interfaces.append(new_interface)
            else:
                configs.append(conf)