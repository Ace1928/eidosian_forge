from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def is_exist_out_direct(self, out_direct, channel_id):
    """if channel out direct exist"""
    if not self.channel_direct_info:
        return False
    for id2name in self.channel_direct_info['channelDirectInfos']:
        if id2name['icOutDirect'] == out_direct and id2name['icCfgChnlId'] == channel_id:
            return True
    return False