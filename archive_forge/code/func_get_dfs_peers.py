from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def get_dfs_peers(config):
    """get evn peer ip list"""
    get = re.findall('peer ([0-9]+.[0-9]+.[0-9]+.[0-9]+)\\s?(vpn-instance)?\\s?(\\S*)', config)
    if not get:
        return None
    else:
        peers = list()
        for item in get:
            peers.append(dict(ip=item[0], vpn=item[2]))
        return peers