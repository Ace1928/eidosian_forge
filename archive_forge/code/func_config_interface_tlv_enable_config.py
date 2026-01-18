from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def config_interface_tlv_enable_config(self):
    if self.function_lldp_interface_flag == 'tlvenableINTERFACE':
        if self.enable_flag == 1 and self.conf_tlv_enable_exsit:
            if self.type_tlv_enable == 'dot1_tlv':
                if self.ifname:
                    if self.protoidtxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_ENABLE_PROTOIDTXENABLE % self.protoidtxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_ENABLE_DOT1_PORT_VLAN')
                        self.changed = True
            if self.type_tlv_enable == 'dcbx':
                if self.ifname:
                    if self.dcbx:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_ENABLE_DCBX % self.dcbx + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_ENABLE_DCBX_VLAN')
                        self.changed = True