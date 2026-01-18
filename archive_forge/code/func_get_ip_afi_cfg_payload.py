from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_ip_afi_cfg_payload(ip_afi):
    ip_afi_cfg = {}
    if ip_afi.get('default_policy_name', None) is not None:
        default_policy_name = ip_afi['default_policy_name']
        ip_afi_cfg.update({'default-policy-name': default_policy_name})
    if ip_afi.get('send_default_route', None) is not None:
        send_default_route = ip_afi['send_default_route']
        ip_afi_cfg.update({'send-default-route': send_default_route})
    return ip_afi_cfg