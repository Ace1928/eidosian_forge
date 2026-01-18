from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_dfs_group(self):
    """delete dfg group"""
    conf_str = CE_NC_DELETE_DFS_GROUP_INFO_HEADER % self.dfs_group_id
    conf_str += CE_NC_DELETE_DFS_GROUP_INFO_TAIL
    recv_xml = set_nc_config(self.module, conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Delete DFS group id failed.')
    self.updates_cmd.append('undo dfs-group 1')
    self.changed = True