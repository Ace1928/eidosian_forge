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
def build_delete_request(self, c_attr, h_attr, intf_name, attr):
    method = DELETE
    attributes_payload = {'speed': 'port-speed', 'auto_negotiate': 'auto-negotiate', 'fec': 'openconfig-if-ethernet-ext2:port-fec', 'advertised_speed': 'openconfig-if-ethernet-ext2:advertised-speed'}
    config_url = (url + eth_conf_url) % quote(intf_name, safe='')
    payload = {'openconfig-if-ethernet:config': {}}
    payload_attr = attributes_payload.get(attr, attr)
    if attr in ('description', 'mtu', 'enabled'):
        attr_url = '/config/' + payload_attr
        config_url = (url + attr_url) % quote(intf_name, safe='')
        return {'path': config_url, 'method': method}
    elif attr in 'fec':
        payload_attr = attributes_payload[attr]
        payload['openconfig-if-ethernet:config'][payload_attr] = 'FEC_DISABLED'
        return {'path': config_url, 'method': PATCH, 'data': payload}
    else:
        payload_attr = attributes_payload[attr]
        if attr == 'auto_negotiate':
            payload['openconfig-if-ethernet:config'][payload_attr] = False
            return {'path': config_url, 'method': PATCH, 'data': payload}
        if attr == 'speed':
            attr_url = eth_conf_url + '/' + attributes_payload[attr]
            del_config_url = (url + attr_url) % quote(intf_name, safe='')
            return {'path': del_config_url, 'method': method}
        if attr == 'advertised_speed':
            new_ads = list(set(h_attr).difference(c_attr))
            if new_ads:
                payload['openconfig-if-ethernet:config'][payload_attr] = ','.join(new_ads)
                return {'path': config_url, 'method': PATCH, 'data': payload}
            else:
                attr_url = eth_conf_url + '/' + attributes_payload[attr]
                del_config_url = (url + attr_url) % quote(intf_name, safe='')
                return {'path': del_config_url, 'method': method}
    return {}