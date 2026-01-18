from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_trap_global(self):
    """ Merge trap global """
    conf_str = CE_MERGE_TRAP_GLOBAL_HEADER
    if self.trap_time_stamp:
        conf_str += '<trapTimeStamp>%s</trapTimeStamp>' % self.trap_time_stamp.upper()
    if self.trap_buff_enable != 'no_use':
        conf_str += '<icTrapBuffEn>%s</icTrapBuffEn>' % self.trap_buff_enable
    if self.trap_buff_size:
        conf_str += '<trapBuffSize>%s</trapBuffSize>' % self.trap_buff_size
    conf_str += CE_MERGE_TRAP_GLOBAL_TAIL
    recv_xml = self.netconf_set_config(conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Merge trap global failed.')
    if self.trap_time_stamp:
        cmd = 'info-center timestamp trap ' + TIME_STAMP_DICT.get(self.trap_time_stamp)
        self.updates_cmd.append(cmd)
    if self.trap_buff_enable != 'no_use':
        if self.trap_buff_enable == 'true':
            cmd = 'info-center trapbuffer'
        else:
            cmd = 'undo info-center trapbuffer'
        self.updates_cmd.append(cmd)
    if self.trap_buff_size:
        cmd = 'info-center trapbuffer size %s' % self.trap_buff_size
        self.updates_cmd.append(cmd)
    self.changed = True