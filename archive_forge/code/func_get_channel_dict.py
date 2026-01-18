from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def get_channel_dict(self):
    """ get channel attributes dict."""
    channel_info = dict()
    conf_str = CE_NC_GET_CHANNEL_INFO % self.channel_id
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return channel_info
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    channel_info['channelInfos'] = list()
    channels = root.findall('syslog/icChannels/icChannel')
    if channels:
        for channel in channels:
            channel_dict = dict()
            for ele in channel:
                if ele.tag in ['icChnlId', 'icChnlCfgName']:
                    channel_dict[ele.tag] = ele.text
            channel_info['channelInfos'].append(channel_dict)
    return channel_info