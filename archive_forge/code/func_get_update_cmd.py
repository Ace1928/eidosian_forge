from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config, execute_nc_action
def get_update_cmd(self):
    """Get updated commands"""
    update_list = list()
    if self.state == 'present':
        if self.lldpenable == 'enabled':
            cli_str = 'lldp enable'
            update_list.append(cli_str)
            if self.ifname:
                cli_str = '%s %s' % ('interface', self.ifname)
                update_list.append(cli_str)
            if self.mdnstatus:
                if self.mdnstatus == 'rxOnly':
                    cli_str = 'lldp mdn enable'
                    update_list.append(cli_str)
                else:
                    cli_str = 'undo lldp mdn enable'
                    update_list.append(cli_str)
        elif self.lldpenable == 'disabled':
            cli_str = 'undo lldp enable'
            update_list.append(cli_str)
        elif self.enable_flag == 1:
            if self.ifname:
                cli_str = '%s %s' % ('interface', self.ifname)
                update_list.append(cli_str)
            if self.mdnstatus:
                if self.mdnstatus == 'rxOnly':
                    cli_str = 'lldp mdn enable'
                    update_list.append(cli_str)
                else:
                    cli_str = 'undo lldp mdn enable'
                    update_list.append(cli_str)
    self.updates_cmd.append(update_list)