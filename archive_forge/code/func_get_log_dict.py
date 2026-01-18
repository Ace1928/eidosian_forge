from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_log_dict(self):
    """ log config dict"""
    log_dict = dict()
    if self.module_name:
        if self.module_name.lower() == 'default':
            conf_str = CE_NC_GET_LOG % (self.module_name.lower(), self.channel_id)
        else:
            conf_str = CE_NC_GET_LOG % (self.module_name.upper(), self.channel_id)
    else:
        conf_str = CE_NC_GET_LOG_GLOBAL
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return log_dict
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    glb = root.find('syslog/globalParam')
    if glb:
        for attr in glb:
            if attr.tag in ['bufferSize', 'logTimeStamp', 'icLogBuffEn']:
                log_dict[attr.tag] = attr.text
    log_dict['source'] = dict()
    src = root.find('syslog/icSources/icSource')
    if src:
        for attr in src:
            if attr.tag in ['moduleName', 'icChannelId', 'icChannelName', 'logEnFlg', 'logEnLevel']:
                log_dict['source'][attr.tag] = attr.text
    return log_dict