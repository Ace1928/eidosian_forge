from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_connection, rm_config_prefix
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def present_netstream(self):
    """ Present netstream configuration """
    cmds = list()
    need_create_record = False
    if self.type == 'ip':
        cmd = 'netstream record %s ip' % self.record_name
    else:
        cmd = 'netstream record %s vxlan inner-ip' % self.record_name
    cmds.append(cmd)
    if self.existing.get('record_name') != self.record_name:
        self.updates_cmd.append(cmd)
        need_create_record = True
    if self.description:
        cmd = 'description %s' % self.description.strip()
        if need_create_record or not self.netstream_cfg or cmd not in self.netstream_cfg:
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
    if self.match:
        if self.type == 'ip':
            cmd = 'match ip %s' % self.match
            cfg = 'match ip'
        else:
            cmd = 'match inner-ip %s' % self.match
            cfg = 'match inner-ip'
        if need_create_record or cfg not in self.netstream_cfg or self.match != self.existing['match'][0]:
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
    if self.collect_counter:
        cmd = 'collect counter %s' % self.collect_counter
        if need_create_record or cmd not in self.netstream_cfg:
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
    if self.collect_interface:
        cmd = 'collect interface %s' % self.collect_interface
        if need_create_record or cmd not in self.netstream_cfg:
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
    if cmds:
        self.cli_load_config(cmds)
        self.changed = True