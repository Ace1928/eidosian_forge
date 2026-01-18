from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def set_mlag_global(self):
    """set mlag global attribute info"""
    if self.is_mlag_global_info_change():
        conf_str = CE_NC_SET_GLOBAL_LACP_MLAG_INFO_HEAD
        if self.mlag_priority_id:
            conf_str += '<lacpMlagPriority>%s</lacpMlagPriority>' % self.mlag_priority_id
        if self.mlag_system_id:
            conf_str += '<lacpMlagSysId>%s</lacpMlagSysId>' % self.mlag_system_id
        conf_str += CE_NC_SET_GLOBAL_LACP_MLAG_INFO_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: set mlag interface attribute info failed.')
        if self.mlag_priority_id:
            self.updates_cmd.append('lacp m-lag priority %s' % self.mlag_priority_id)
        if self.mlag_system_id:
            self.updates_cmd.append('lacp m-lag system-id %s' % self.mlag_system_id)
        self.changed = True