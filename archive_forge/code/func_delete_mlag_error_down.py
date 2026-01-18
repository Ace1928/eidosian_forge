from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_mlag_error_down(self):
    """delete mlag error down info"""
    if self.is_mlag_error_down_info_exist():
        conf_str = CE_NC_DELETE_MLAG_ERROR_DOWN_INFO % self.interface
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: delete mlag error down info failed.')
        self.updates_cmd.append('interface %s' % self.interface)
        self.updates_cmd.append('undo m-lag unpaired-port suspend')
        self.changed = True