from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def get_lldp_exist_config(self):
    """Get lldp existed configure"""
    lldp_config = list()
    lldp_dict = dict()
    conf_enable_str = CE_NC_GET_GLOBAL_LLDPENABLE_CONFIG
    conf_enable_obj = get_nc_config(self.module, conf_enable_str)
    xml_enable_str = conf_enable_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root_enable = ElementTree.fromstring(xml_enable_str)
    ntpsite_enable = root_enable.findall('lldp/lldpSys')
    for nexthop_enable in ntpsite_enable:
        for ele_enable in nexthop_enable:
            if ele_enable.tag in ['lldpEnable', 'mdnStatus']:
                lldp_dict[ele_enable.tag] = ele_enable.text
        if self.state == 'present':
            cur_lldp_cfg = dict(lldpenable=lldp_dict['lldpEnable'], mdnstatus=lldp_dict['mdnStatus'])
            exp_lldp_cfg = dict(lldpenable=self.lldpenable, mdnstatus=self.mdnstatus)
            if lldp_dict['lldpEnable'] == 'enabled':
                self.enable_flag = 1
            if cur_lldp_cfg == exp_lldp_cfg:
                self.conf_exsit = True
        lldp_config.append(dict(lldpenable=lldp_dict['lldpEnable'], mdnstatus=lldp_dict['mdnStatus']))
    conf_str = CE_NC_GET_GLOBAL_LLDP_CONFIG
    conf_obj = get_nc_config(self.module, conf_str)
    if '<data/>' in conf_obj:
        pass
    else:
        xml_str = conf_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        ntpsite = root.findall('lldp/lldpSys/lldpSysParameter')
        for nexthop in ntpsite:
            for ele in nexthop:
                if ele.tag in ['messageTxInterval', 'messageTxHoldMultiplier', 'reinitDelay', 'txDelay', 'notificationInterval', 'fastMessageCount', 'mdnNotificationInterval', 'configManAddr', 'bindifName']:
                    lldp_dict[ele.tag] = ele.text
            if self.state == 'present':
                cur_ntp_cfg = dict(interval=lldp_dict['messageTxInterval'], hold_multiplier=lldp_dict['messageTxHoldMultiplier'], restart_delay=lldp_dict['reinitDelay'], transmit_delay=lldp_dict['txDelay'], notification_interval=lldp_dict['notificationInterval'], fast_count=lldp_dict['fastMessageCount'], mdn_notification_interval=lldp_dict['mdnNotificationInterval'], management_address=lldp_dict['configManAddr'], bind_name=lldp_dict['bindifName'])
                exp_ntp_cfg = dict(interval=self.interval, hold_multiplier=self.hold_multiplier, restart_delay=self.restart_delay, transmit_delay=self.transmit_delay, notification_interval=self.notification_interval, fast_count=self.fast_count, mdn_notification_interval=self.mdn_notification_interval, management_address=self.management_address, bind_name=self.bind_name)
                if cur_ntp_cfg == exp_ntp_cfg:
                    self.conf_exsit_lldp = True
            lldp_config.append(dict(interval=lldp_dict['messageTxInterval'], hold_multiplier=lldp_dict['messageTxHoldMultiplier'], restart_delay=lldp_dict['reinitDelay'], transmit_delay=lldp_dict['txDelay'], notification_interval=lldp_dict['notificationInterval'], fast_count=lldp_dict['fastMessageCount'], mdn_notification_interval=lldp_dict['mdnNotificationInterval'], management_address=lldp_dict['configManAddr'], bind_name=lldp_dict['bindifName']))
    tmp_dict = dict()
    str_1 = str(lldp_config)
    temp_1 = str_1.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace("'", '')
    if temp_1:
        tmp_2 = temp_1.split(',')
        for i in tmp_2:
            tmp_value = re.match('(.*):(.*)', i)
            key_tmp = tmp_value.group(1)
            key_value = tmp_value.group(2)
            tmp_dict[key_tmp] = key_value
    return tmp_dict