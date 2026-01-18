from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def get_interface_tlv_disable_config(self):
    lldp_config = list()
    lldp_dict = dict()
    cur_interface_mdn_cfg = dict()
    exp_interface_mdn_cfg = dict()
    if self.enable_flag == 1:
        conf_str = CE_NC_GET_INTERFACE_TLV_DISABLE_CONFIG
        conf_obj = get_nc_config(self.module, conf_str)
        if '<data/>' in conf_obj:
            return lldp_config
        xml_str = conf_obj.replace('\r', '').replace('\n', '')
        xml_str = xml_str.replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '')
        xml_str = xml_str.replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        lldp_tlvdisable_ifname = root.findall('lldp/lldpInterfaces/lldpInterface')
        for ele in lldp_tlvdisable_ifname:
            ifname_tmp = ele.find('ifName')
            manaddrtxenable_tmp = ele.find('tlvTxEnable/manAddrTxEnable')
            portdesctxenable_tmp = ele.find('tlvTxEnable/portDescTxEnable')
            syscaptxenable_tmp = ele.find('tlvTxEnable/sysCapTxEnable')
            sysdesctxenable_tmp = ele.find('tlvTxEnable/sysDescTxEnable')
            sysnametxenable_tmp = ele.find('tlvTxEnable/sysNameTxEnable')
            linkaggretxenable_tmp = ele.find('tlvTxEnable/linkAggreTxEnable')
            macphytxenable_tmp = ele.find('tlvTxEnable/macPhyTxEnable')
            maxframetxenable_tmp = ele.find('tlvTxEnable/maxFrameTxEnable')
            eee_tmp = ele.find('tlvTxEnable/eee')
            if ifname_tmp is not None:
                if ifname_tmp.text is not None:
                    cur_interface_mdn_cfg['ifname'] = ifname_tmp.text
            if ifname_tmp is not None and manaddrtxenable_tmp is not None:
                if manaddrtxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['manaddrtxenable'] = manaddrtxenable_tmp.text
            if ifname_tmp is not None and portdesctxenable_tmp is not None:
                if portdesctxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['portdesctxenable'] = portdesctxenable_tmp.text
            if ifname_tmp is not None and syscaptxenable_tmp is not None:
                if syscaptxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['syscaptxenable'] = syscaptxenable_tmp.text
            if ifname_tmp is not None and sysdesctxenable_tmp is not None:
                if sysdesctxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['sysdesctxenable'] = sysdesctxenable_tmp.text
            if ifname_tmp is not None and sysnametxenable_tmp is not None:
                if sysnametxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['sysnametxenable'] = sysnametxenable_tmp.text
            if ifname_tmp is not None and linkaggretxenable_tmp is not None:
                if linkaggretxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['linkaggretxenable'] = linkaggretxenable_tmp.text
            if ifname_tmp is not None and macphytxenable_tmp is not None:
                if macphytxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['macphytxenable'] = macphytxenable_tmp.text
            if ifname_tmp is not None and maxframetxenable_tmp is not None:
                if maxframetxenable_tmp.text is not None:
                    cur_interface_mdn_cfg['maxframetxenable'] = maxframetxenable_tmp.text
            if ifname_tmp is not None and eee_tmp is not None:
                if eee_tmp.text is not None:
                    cur_interface_mdn_cfg['eee'] = eee_tmp.text
            if self.state == 'present':
                if self.function_lldp_interface_flag == 'tlvdisableINTERFACE':
                    if self.type_tlv_disable == 'basic_tlv':
                        if self.ifname:
                            exp_interface_mdn_cfg['ifname'] = self.ifname
                            if self.manaddrtxenable:
                                exp_interface_mdn_cfg['manaddrtxenable'] = self.manaddrtxenable
                            if self.portdesctxenable:
                                exp_interface_mdn_cfg['portdesctxenable'] = self.portdesctxenable
                            if self.syscaptxenable:
                                exp_interface_mdn_cfg['syscaptxenable'] = self.syscaptxenable
                            if self.sysdesctxenable:
                                exp_interface_mdn_cfg['sysdesctxenable'] = self.sysdesctxenable
                            if self.sysnametxenable:
                                exp_interface_mdn_cfg['sysnametxenable'] = self.sysnametxenable
                            if self.ifname == ifname_tmp.text:
                                key_list = exp_interface_mdn_cfg.keys()
                                key_list_cur = cur_interface_mdn_cfg.keys()
                                if len(key_list) != 0:
                                    for key in key_list:
                                        if key == 'ifname' and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(ifname=cur_interface_mdn_cfg['ifname']))
                                        if 'manaddrtxenable' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(manaddrtxenable=cur_interface_mdn_cfg['manaddrtxenable']))
                                        if 'portdesctxenable' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(portdesctxenable=cur_interface_mdn_cfg['portdesctxenable']))
                                        if 'syscaptxenable' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(syscaptxenable=cur_interface_mdn_cfg['syscaptxenable']))
                                        if 'sysdesctxenable' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(sysdesctxenable=cur_interface_mdn_cfg['sysdesctxenable']))
                                        if 'sysnametxenable' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(sysnametxenable=cur_interface_mdn_cfg['sysnametxenable']))
                                        if key in key_list_cur:
                                            if str(exp_interface_mdn_cfg[key]) != str(cur_interface_mdn_cfg[key]):
                                                self.conf_tlv_disable_exsit = True
                                                self.changed = True
                                                return lldp_config
                                        else:
                                            self.conf_tlv_disable_exsit = True
                                            return lldp_config
                    if self.type_tlv_disable == 'dot3_tlv':
                        if self.ifname:
                            exp_interface_mdn_cfg['ifname'] = self.ifname
                            if self.linkaggretxenable:
                                exp_interface_mdn_cfg['linkaggretxenable'] = self.linkaggretxenable
                            if self.macphytxenable:
                                exp_interface_mdn_cfg['macphytxenable'] = self.macphytxenable
                            if self.maxframetxenable:
                                exp_interface_mdn_cfg['maxframetxenable'] = self.maxframetxenable
                            if self.eee:
                                exp_interface_mdn_cfg['eee'] = self.eee
                            if self.ifname == ifname_tmp.text:
                                key_list = exp_interface_mdn_cfg.keys()
                                key_list_cur = cur_interface_mdn_cfg.keys()
                                if len(key_list) != 0:
                                    for key in key_list:
                                        if key == 'ifname' and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(ifname=cur_interface_mdn_cfg['ifname']))
                                        if 'linkaggretxenable' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(linkaggretxenable=cur_interface_mdn_cfg['linkaggretxenable']))
                                        if 'macphytxenable' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(macphytxenable=cur_interface_mdn_cfg['macphytxenable']))
                                        if 'maxframetxenable' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(maxframetxenable=cur_interface_mdn_cfg['maxframetxenable']))
                                        if 'eee' == key and self.ifname == cur_interface_mdn_cfg['ifname']:
                                            lldp_config.append(dict(eee=cur_interface_mdn_cfg['eee']))
                                        if key in key_list_cur:
                                            if str(exp_interface_mdn_cfg[key]) != str(cur_interface_mdn_cfg[key]):
                                                self.conf_tlv_disable_exsit = True
                                                self.changed = True
                                                return lldp_config
                                        else:
                                            self.conf_tlv_disable_exsit = True
                                            return lldp_config
    return lldp_config