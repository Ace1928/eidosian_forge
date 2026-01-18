from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def config_lldp(self):
    """Configure lldp enabled and mdn enabled parameters"""
    if self.state == 'present':
        if (self.enable_flag == 1 and self.lldpenable == 'enabled') and (not self.conf_exsit):
            if self.mdnstatus:
                xml_str = CE_NC_MERGE_GLOBA_MDNENABLE_CONFIG % self.mdnstatus
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'MDN_ENABLE_CONFIG')
        if self.lldpenable == 'enabled' and (not self.conf_exsit):
            xml_str = CE_NC_MERGE_GLOBA_LLDPENABLE_CONFIG % self.lldpenable
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'LLDP_ENABLE_CONFIG')
            if self.mdnstatus:
                xml_str = CE_NC_MERGE_GLOBA_MDNENABLE_CONFIG % self.mdnstatus
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'MDN_ENABLE_CONFIG')
        if self.enable_flag == 1 and (not self.conf_exsit):
            if self.mdnstatus:
                xml_str = CE_NC_MERGE_GLOBA_MDNENABLE_CONFIG % self.mdnstatus
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'MDN_ENABLE_CONFIG')
        if (self.lldpenable == 'enabled' or self.enable_flag == 1) and (not self.conf_exsit_lldp):
            if self.hold_multiplier:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HOLD_MULTIPLIER % self.hold_multiplier + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.interval:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_INTERVAL % self.interval + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.restart_delay:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_RESTART_DELAY % self.restart_delay + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.transmit_delay:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TRANSMIT_DELAY % self.transmit_delay + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.notification_interval:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_NOTIFICATION_INTERVAL % self.notification_interval + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.fast_count:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_FAST_COUNT % self.fast_count + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.mdn_notification_interval:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_MDN_NOTIFICATION_INTERVAL % self.mdn_notification_interval + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.management_address:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_MANAGEMENT_ADDRESS % self.management_address + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.bind_name:
                xml_str = CE_NC_MERGE_GLOBAL_LLDP_CONFIG_HEADER + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_BIND_NAME % self.bind_name + CE_NC_MERGE_GLOBAL_LLDP_CONFIG_TAIL
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_CONFIG_INTERVAL')
            if self.lldpenable == 'disabled' and (not self.conf_exsit):
                xml_str = CE_NC_MERGE_GLOBA_LLDPENABLE_CONFIG % self.lldpenable
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'LLDP_DISABLE_CONFIG')