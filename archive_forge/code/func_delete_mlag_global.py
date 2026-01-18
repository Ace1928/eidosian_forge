from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_mlag_global(self):
    """delete mlag global attribute info"""
    xml_str = ''
    if self.is_mlag_global_info_exist():
        if self.mlag_priority_id:
            cmd = 'lacp m-lag priority %s' % self.mlag_priority_id
            xml_str += '<lacpMlagPriority></lacpMlagPriority>'
            self.cli_add_command(cmd, True)
        if self.mlag_system_id:
            cmd = 'lacp m-lag system-id %s' % self.mlag_system_id
            xml_str += '<lacpMlagSysId></lacpMlagSysId>'
            self.cli_add_command(cmd, True)
        if xml_str != '':
            conf_str = CE_NC_SET_GLOBAL_LACP_MLAG_INFO_HEAD + xml_str + CE_NC_SET_GLOBAL_LACP_MLAG_INFO_TAIL
            recv_xml = set_nc_config(self.module, conf_str)
            if '<ok/>' not in recv_xml:
                self.module.fail_json(msg='Error: set mlag interface atrribute info failed.')
            self.changed = True