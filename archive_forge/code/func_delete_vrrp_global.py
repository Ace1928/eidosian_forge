from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_vrrp_global(self):
    """delete vrrp global attribute info"""
    if self.is_vrrp_global_info_exist():
        conf_str = CE_NC_SET_VRRP_GLOBAL_HEAD
        if self.gratuitous_arp_interval:
            if self.gratuitous_arp_interval == '120':
                self.module.fail_json(msg='Error: The default value of gratuitous_arp_interval is 120.')
            gratuitous_arp_interval = '120'
            conf_str += '<gratuitousArpTimeOut>%s</gratuitousArpTimeOut>' % gratuitous_arp_interval
        if self.recover_delay:
            if self.recover_delay == '0':
                self.module.fail_json(msg='Error: The default value of recover_delay is 0.')
            recover_delay = '0'
            conf_str += '<recoverDelay>%s</recoverDelay>' % recover_delay
        if self.version:
            if self.version == 'v2':
                self.module.fail_json(msg='Error: The default value of version is v2.')
            version = 'v2'
            conf_str += '<version>%s</version>' % version
        conf_str += CE_NC_SET_VRRP_GLOBAL_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: set vrrp global attribute info failed.')
        if self.gratuitous_arp_interval:
            self.updates_cmd.append('undo vrrp gratuitous-arp interval')
        if self.recover_delay:
            self.updates_cmd.append('undo vrrp recover-delay')
        if self.version == 'v3':
            self.updates_cmd.append('undo vrrp version')
        self.changed = True