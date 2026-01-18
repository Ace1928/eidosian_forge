from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def get_peers_enable(self):
    """get evpn peer address enable list"""
    if len(self.config_list) != 2:
        return None
    self.config_list = self.config.split('l2vpn-family evpn')
    get = re.findall('peer ([0-9]+.[0-9]+.[0-9]+.[0-9]+)\\s?as-number\\s?(\\S*)', self.config_list[0])
    if not get:
        return None
    else:
        peers = list()
        for item in get:
            cmd = 'peer %s enable' % item[0]
            exist = is_config_exist(self.config_list[1], cmd)
            if exist:
                peers.append(dict(peer_address=item[0], as_number=item[1], peer_enable='true'))
            else:
                peers.append(dict(peer_address=item[0], as_number=item[1], peer_enable='false'))
        return peers