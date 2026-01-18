from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_mlag_info_change(self):
    """whether mlag info change"""
    if not self.mlag_info:
        return True
    eth_trunk = 'Eth-Trunk'
    eth_trunk += self.eth_trunk_id
    for info in self.mlag_info['mlagInfos']:
        if info['mlagId'] == self.mlag_id and info['localMlagPort'] == eth_trunk:
            return False
    return True