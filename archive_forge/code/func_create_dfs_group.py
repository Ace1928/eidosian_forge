from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def create_dfs_group(self):
    """create dfs group info"""
    conf_str = CE_NC_CREATE_DFS_GROUP_INFO_HEADER % self.dfs_group_id
    if self.priority_id and self.priority_id != 100:
        conf_str += '<priority>%s</priority>' % self.priority_id
    if self.ip_address:
        conf_str += '<ipAddress>%s</ipAddress>' % self.ip_address
    if self.vpn_instance_name:
        if not self.ip_address:
            self.module.fail_json(msg='Error: ip_address can not be null if vpn_instance_name is exist.')
        conf_str += '<srcVpnName>%s</srcVpnName>' % self.vpn_instance_name
    if self.nickname or self.pseudo_nickname or self.pseudo_priority:
        conf_str += '<trillType>'
        if self.nickname:
            conf_str += '<localNickname>%s</localNickname>' % self.nickname
        if self.pseudo_nickname:
            conf_str += '<pseudoNickname>%s</pseudoNickname>' % self.pseudo_nickname
        if self.pseudo_priority:
            if not self.pseudo_nickname:
                self.module.fail_json(msg='Error: pseudo_nickname can not be null if pseudo_priority is exist.')
            conf_str += '<pseudoPriority>%s</pseudoPriority>' % self.pseudo_priority
        conf_str += '</trillType>'
    conf_str += CE_NC_CREATE_DFS_GROUP_INFO_TAIL
    recv_xml = set_nc_config(self.module, conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Merge DFS group info failed.')
    self.updates_cmd.append('dfs-group 1')
    if self.priority_id:
        self.updates_cmd.append('priority %s' % self.priority_id)
    if self.ip_address:
        if self.vpn_instance_name:
            self.updates_cmd.append('source ip %s vpn-instance %s' % (self.ip_address, self.vpn_instance_name))
        else:
            self.updates_cmd.append('source ip %s' % self.ip_address)
    if self.nickname:
        self.updates_cmd.append('source nickname %s' % self.nickname)
    if self.pseudo_nickname:
        if self.pseudo_priority:
            self.updates_cmd.append('pseudo-nickname %s priority %s' % (self.pseudo_nickname, self.pseudo_priority))
        else:
            self.updates_cmd.append('pseudo-nickname %s' % self.pseudo_nickname)
    self.changed = True