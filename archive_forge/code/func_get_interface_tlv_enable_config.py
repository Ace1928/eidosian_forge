from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def get_interface_tlv_enable_config(self):
    lldp_config = list()
    lldp_dict = dict()
    cur_interface_mdn_cfg = dict()
    exp_interface_mdn_cfg = dict()
    if self.enable_flag == 1:
        conf_str = CE_NC_GET_INTERFACE_TLV_ENABLE_CONFIG
        conf_obj = get_nc_config(self.module, conf_str)
        if '<data/>' in conf_obj:
            return lldp_config
        xml_str = conf_obj.replace('\r', '').replace('\n', '')
        xml_str = xml_str.replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '')
        xml_str = xml_str.replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        lldpenablesite = root.findall('lldp/lldpInterfaces/lldpInterface')
        for ele in lldpenablesite:
            ifname_tmp = ele.find('ifName')
            protoidtxenable_tmp = ele.find('tlvTxEnable/protoIdTxEnable')
            dcbx_tmp = ele.find('tlvTxEnable/dcbx')
            if ifname_tmp is not None:
                if ifname_tmp.text is not None:
                    cur_interface_mdn_cfg['ifname'] = ifname_tmp.text
            if ifname_tmp is not None and protoidtxenable_tmp is not None:
                if protoidtxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['protoidtxenable'] = protoidtxenable_tmp.text
            if ifname_tmp is not None and dcbx_tmp is not None:
                if dcbx_tmp.text is not None:
                    cur_interface_mdn_cfg['dcbx'] = dcbx_tmp.text
            if self.state == 'present':
                if self.function_lldp_interface_flag == 'tlvenableINTERFACE':
                    if self.type_tlv_enable == 'dot1_tlv':
                        if self.ifname:
                            exp_interface_mdn_cfg['ifname'] = self.ifname
                            if self.protoidtxenable:
                                exp_interface_mdn_cfg['protoidtxenable'] = self.protoidtxenable
                            if self.ifname == ifname_tmp.text:
                                key_list = exp_interface_mdn_cfg.keys()
                                key_list_cur = cur_interface_mdn_cfg.keys()
                                if len(key_list) != 0:
                                    for key in key_list:
                                        if 'protoidtxenable' == str(key) and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(protoidtxenable=cur_interface_mdn_cfg['protoidtxenable']))
                                        if key in key_list_cur:
                                            if str(exp_interface_mdn_cfg[key]) != str(cur_interface_mdn_cfg[key]):
                                                self.conf_tlv_enable_exsit = True
                                                self.changed = True
                                                return lldp_config
                                        else:
                                            self.conf_tlv_enable_exsit = True
                                            return lldp_config
                    if self.type_tlv_enable == 'dcbx':
                        if self.ifname:
                            exp_interface_mdn_cfg['ifname'] = self.ifname
                            if self.dcbx:
                                exp_interface_mdn_cfg['dcbx'] = self.dcbx
                            if self.ifname == ifname_tmp.text:
                                key_list = exp_interface_mdn_cfg.keys()
                                key_list_cur = cur_interface_mdn_cfg.keys()
                                if len(key_list) != 0:
                                    for key in key_list:
                                        if 'dcbx' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(dcbx=cur_interface_mdn_cfg['dcbx']))
                                        if key in key_list_cur:
                                            if str(exp_interface_mdn_cfg[key]) != str(cur_interface_mdn_cfg[key]):
                                                self.conf_tlv_enable_exsit = True
                                                self.changed = True
                                                return lldp_config
                                        else:
                                            self.conf_tlv_enable_exsit = True
                                            return lldp_config
    return lldp_config