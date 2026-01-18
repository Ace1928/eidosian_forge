from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, set_nc_config, get_nc_config, execute_nc_action
def config_intf_dldp(self):
    """Config global dldp"""
    if self.same_conf:
        return
    if self.state == 'present':
        enable = self.enable
        if not self.enable:
            enable = self.dldp_intf_conf['dldpEnable']
        if enable == 'enable':
            enable = 'true'
        else:
            enable = 'false'
        mode_enable = self.mode_enable
        if not self.mode_enable:
            mode_enable = self.dldp_intf_conf['dldpCompatibleEnable']
        if mode_enable == 'enable':
            mode_enable = 'true'
        else:
            mode_enable = 'false'
        local_mac = self.local_mac
        if not self.local_mac:
            local_mac = self.dldp_intf_conf['dldpLocalMac']
        if self.enable == 'disable' and self.enable != self.dldp_intf_conf['dldpEnable']:
            xml_str = CE_NC_DELETE_DLDP_INTF_CONFIG % self.interface
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'DELETE_DLDP_INTF_CONFIG')
        elif self.dldp_intf_conf['dldpEnable'] == 'disable' and self.enable == 'enable':
            xml_str = CE_NC_CREATE_DLDP_INTF_CONFIG % (self.interface, 'true', mode_enable, local_mac)
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'CREATE_DLDP_INTF_CONFIG')
        elif self.dldp_intf_conf['dldpEnable'] == 'enable':
            if mode_enable == 'false':
                local_mac = ''
            xml_str = CE_NC_MERGE_DLDP_INTF_CONFIG % (self.interface, enable, mode_enable, local_mac)
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'MERGE_DLDP_INTF_CONFIG')
        if self.reset == 'enable':
            xml_str = CE_NC_ACTION_RESET_INTF_DLDP % self.interface
            ret_xml = execute_nc_action(self.module, xml_str)
            self.check_response(ret_xml, 'ACTION_RESET_INTF_DLDP')
        self.changed = True
    elif self.local_mac and judge_is_mac_same(self.local_mac, self.dldp_intf_conf['dldpLocalMac']):
        if self.dldp_intf_conf['dldpEnable'] == 'enable':
            dldp_enable = 'true'
        else:
            dldp_enable = 'false'
        if self.dldp_intf_conf['dldpCompatibleEnable'] == 'enable':
            dldp_compat_enable = 'true'
        else:
            dldp_compat_enable = 'false'
        xml_str = CE_NC_MERGE_DLDP_INTF_CONFIG % (self.interface, dldp_enable, dldp_compat_enable, '')
        ret_xml = set_nc_config(self.module, xml_str)
        self.check_response(ret_xml, 'UNDO_DLDP_INTF_LOCAL_MAC_CONFIG')
        self.changed = True