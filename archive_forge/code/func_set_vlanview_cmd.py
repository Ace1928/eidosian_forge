from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_vlanview_cmd(self):
    """set vlanview update command"""
    if not self.changed:
        return
    if self.state == 'present':
        if self.igmp:
            if self.igmp.lower() == 'true':
                self.updates_cmd.append('igmp snooping enable')
            else:
                self.updates_cmd.append('undo igmp snooping enable')
        if str(self.version):
            self.updates_cmd.append('igmp snooping version %s' % self.version)
        else:
            self.updates_cmd.append('undo igmp snooping version')
        if self.proxy:
            if self.proxy.lower() == 'true':
                self.updates_cmd.append('igmp snooping proxy')
            else:
                self.updates_cmd.append('undo igmp snooping proxy')
    else:
        self.updates_cmd.append('undo igmp snooping enable')
        self.updates_cmd.append('undo igmp snooping version')
        self.updates_cmd.append('undo igmp snooping proxy')