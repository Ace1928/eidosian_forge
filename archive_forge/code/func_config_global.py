from __future__ import (absolute_import, division, print_function)
import sys
import socket
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def config_global(self):
    """configures bfd global params"""
    xml_str = ''
    damp_chg = False
    if self.bfd_enable:
        if bool(self.bfd_dict['global'].get('bfdEnable', 'false') == 'true') != bool(self.bfd_enable == 'enable'):
            if self.bfd_enable == 'enable':
                xml_str = '<bfdEnable>true</bfdEnable>'
                self.updates_cmd.append('bfd')
            else:
                xml_str = '<bfdEnable>false</bfdEnable>'
                self.updates_cmd.append('undo bfd')
    bfd_state = 'disable'
    if self.bfd_enable:
        bfd_state = self.bfd_enable
    elif self.bfd_dict['global'].get('bfdEnable', 'false') == 'true':
        bfd_state = 'enable'
    if self.default_ip:
        if bfd_state == 'enable':
            if self.state == 'present' and self.default_ip != self.bfd_dict['global'].get('defaultIp'):
                xml_str += '<defaultIp>%s</defaultIp>' % self.default_ip
                if 'bfd' not in self.updates_cmd:
                    self.updates_cmd.append('bfd')
                self.updates_cmd.append('default-ip-address %s' % self.default_ip)
            elif self.state == 'absent' and self.default_ip == self.bfd_dict['global'].get('defaultIp'):
                xml_str += '<defaultIp/>'
                if 'bfd' not in self.updates_cmd:
                    self.updates_cmd.append('bfd')
                self.updates_cmd.append('undo default-ip-address')
    if self.tos_exp_dynamic is not None:
        if bfd_state == 'enable':
            if self.state == 'present' and self.tos_exp_dynamic != int(self.bfd_dict['global'].get('tosExp', '7')):
                xml_str += '<tosExp>%s</tosExp>' % self.tos_exp_dynamic
                if 'bfd' not in self.updates_cmd:
                    self.updates_cmd.append('bfd')
                self.updates_cmd.append('tos-exp %s dynamic' % self.tos_exp_dynamic)
            elif self.state == 'absent' and self.tos_exp_dynamic == int(self.bfd_dict['global'].get('tosExp', '7')):
                xml_str += '<tosExp/>'
                if 'bfd' not in self.updates_cmd:
                    self.updates_cmd.append('bfd')
                self.updates_cmd.append('undo tos-exp dynamic')
    if self.tos_exp_static is not None:
        if bfd_state == 'enable':
            if self.state == 'present' and self.tos_exp_static != int(self.bfd_dict['global'].get('tosExpStatic', '7')):
                xml_str += '<tosExpStatic>%s</tosExpStatic>' % self.tos_exp_static
                if 'bfd' not in self.updates_cmd:
                    self.updates_cmd.append('bfd')
                self.updates_cmd.append('tos-exp %s static' % self.tos_exp_static)
            elif self.state == 'absent' and self.tos_exp_static == int(self.bfd_dict['global'].get('tosExpStatic', '7')):
                xml_str += '<tosExpStatic/>'
                if 'bfd' not in self.updates_cmd:
                    self.updates_cmd.append('bfd')
                self.updates_cmd.append('undo tos-exp static')
    if self.delay_up_time is not None:
        if bfd_state == 'enable':
            delay_time = self.bfd_dict['global'].get('delayUpTimer', '0')
            if not delay_time or not delay_time.isdigit():
                delay_time = '0'
            if self.state == 'present' and self.delay_up_time != int(delay_time):
                xml_str += '<delayUpTimer>%s</delayUpTimer>' % self.delay_up_time
                if 'bfd' not in self.updates_cmd:
                    self.updates_cmd.append('bfd')
                self.updates_cmd.append('delay-up %s' % self.delay_up_time)
            elif self.state == 'absent' and self.delay_up_time == int(delay_time):
                xml_str += '<delayUpTimer/>'
                if 'bfd' not in self.updates_cmd:
                    self.updates_cmd.append('bfd')
                self.updates_cmd.append('undo delay-up')
    if self.damp_init_wait_time is not None and self.damp_second_wait_time is not None and (self.damp_second_wait_time is not None):
        if bfd_state == 'enable':
            if self.state == 'present':
                if self.damp_max_wait_time != int(self.bfd_dict['global'].get('dampMaxWaitTime', '2000')):
                    xml_str += '<dampMaxWaitTime>%s</dampMaxWaitTime>' % self.damp_max_wait_time
                    damp_chg = True
                if self.damp_init_wait_time != int(self.bfd_dict['global'].get('dampInitWaitTime', '12000')):
                    xml_str += '<dampInitWaitTime>%s</dampInitWaitTime>' % self.damp_init_wait_time
                    damp_chg = True
                if self.damp_second_wait_time != int(self.bfd_dict['global'].get('dampSecondWaitTime', '5000')):
                    xml_str += '<dampSecondWaitTime>%s</dampSecondWaitTime>' % self.damp_second_wait_time
                    damp_chg = True
                if damp_chg:
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('dampening timer-interval maximum %s initial %s secondary %s' % (self.damp_max_wait_time, self.damp_init_wait_time, self.damp_second_wait_time))
            else:
                damp_chg = True
                if self.damp_max_wait_time != int(self.bfd_dict['global'].get('dampMaxWaitTime', '2000')):
                    damp_chg = False
                if self.damp_init_wait_time != int(self.bfd_dict['global'].get('dampInitWaitTime', '12000')):
                    damp_chg = False
                if self.damp_second_wait_time != int(self.bfd_dict['global'].get('dampSecondWaitTime', '5000')):
                    damp_chg = False
                if damp_chg:
                    xml_str += '<dampMaxWaitTime/><dampInitWaitTime/><dampSecondWaitTime/>'
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('undo dampening timer-interval maximum %s initial %s secondary %s' % (self.damp_max_wait_time, self.damp_init_wait_time, self.damp_second_wait_time))
    if xml_str:
        return '<bfdSchGlobal operation="merge">' + xml_str + '</bfdSchGlobal>'
    else:
        return ''