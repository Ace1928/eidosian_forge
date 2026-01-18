from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_sysview_cmd(self):
    """set sysview update command"""
    if not self.changed:
        return
    if self.state == 'present':
        self.updates_cmd.append('igmp snooping enable')
    else:
        self.updates_cmd.append('undo igmp snooping enable')