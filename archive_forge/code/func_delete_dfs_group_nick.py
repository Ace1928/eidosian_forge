from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_dfs_group_nick(self):
    conf_str = CE_NC_DELETE_DFS_GROUP_ATTRIBUTE_HEADER % self.dfs_group_id
    conf_str = conf_str.replace('<groupInstance  operation="delete">', '<groupInstance>')
    change = False
    if self.nickname or self.pseudo_nickname:
        conf_str += "<trillType operation='delete'>"
        if self.nickname and self.dfs_group_info['localNickname'] == self.nickname:
            conf_str += '<localNickname>%s</localNickname>' % self.nickname
            change = True
            self.updates_cmd.append('undo source nickname %s' % self.nickname)
        if self.pseudo_nickname and self.dfs_group_info['pseudoNickname'] == self.pseudo_nickname:
            conf_str += '<pseudoNickname>%s</pseudoNickname>' % self.pseudo_nickname
            if self.pseudo_priority and self.dfs_group_info['pseudoPriority'] == self.pseudo_priority:
                self.updates_cmd.append('undo pseudo-nickname %s priority %s' % (self.pseudo_nickname, self.pseudo_priority))
            if not self.pseudo_priority:
                self.updates_cmd.append('undo pseudo-nickname %s' % self.pseudo_nickname)
            change = True
        conf_str += '</trillType>'
    conf_str += CE_NC_DELETE_DFS_GROUP_ATTRIBUTE_TAIL
    if change:
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Delete DFS group attribute failed.')
        self.changed = True