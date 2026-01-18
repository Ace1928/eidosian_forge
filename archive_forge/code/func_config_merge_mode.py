from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_merge_mode(self, nve_name, mode):
    """config nve mode"""
    if self.is_nve_mode_change(nve_name, mode):
        cfg_xml = CE_NC_MERGE_NVE_MODE % (nve_name, mode)
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'MERGE_MODE')
        self.updates_cmd.append('interface %s' % nve_name)
        if mode == 'mode-l3':
            self.updates_cmd.append('mode l3')
        else:
            self.updates_cmd.append('undo mode l3')
        self.changed = True