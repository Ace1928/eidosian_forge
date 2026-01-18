from __future__ import absolute_import, division, print_function
from natsort import (
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.interfaces_util import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import re
import traceback
def handle_delete_interface_config(self, commands, have, delete_all=False):
    if not commands:
        return ([], [])
    commands_del, requests = ([], [])
    for conf in commands:
        name = conf['name']
        have_conf = next((cfg for cfg in have if cfg['name'] == name), None)
        if have_conf:
            lp_key_set = set(conf.keys())
            if name.startswith('Loopback'):
                if delete_all or len(lp_key_set) == 1:
                    method = DELETE
                    lpbk_url = url % quote(name, safe='')
                    request = {'path': lpbk_url, 'method': DELETE}
                    requests.append(request)
                    commands_del.append({'name': name})
                    continue
            cmd = deepcopy(have_conf) if len(lp_key_set) == 1 else deepcopy(conf)
            del_cmd = {'name': name}
            attribute = eth_attribute if name.startswith('Eth') else non_eth_attribute
            for attr in attribute:
                if attr in conf:
                    c_attr = conf.get(attr)
                    h_attr = have_conf.get(attr)
                    default_val = self.get_default_value(attr, h_attr, name)
                    if c_attr is not None and h_attr is not None and (h_attr != default_val):
                        if attr == 'advertised_speed':
                            c_ads = c_attr if c_attr else []
                            h_ads = h_attr if h_attr else []
                            new_ads = list(set(h_attr).intersection(c_attr))
                            if new_ads:
                                del_cmd.update({attr: new_ads})
                                requests.append(self.build_delete_request(c_ads, h_ads, name, attr))
                        else:
                            del_cmd.update({attr: h_attr})
                            requests.append(self.build_delete_request(c_attr, h_attr, name, attr))
        if requests:
            commands_del.append(del_cmd)
    return (commands_del, requests)