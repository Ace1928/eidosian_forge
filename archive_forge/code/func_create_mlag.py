from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def create_mlag(self):
    """create mlag info"""
    if self.is_mlag_info_change():
        mlag_port = 'Eth-Trunk'
        mlag_port += self.eth_trunk_id
        conf_str = CE_NC_CREATE_MLAG_INFO % (self.dfs_group_id, self.mlag_id, mlag_port)
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: create mlag info failed.')
        self.updates_cmd.append('interface %s' % mlag_port)
        self.updates_cmd.append('dfs-group %s m-lag %s' % (self.dfs_group_id, self.mlag_id))
        self.changed = True