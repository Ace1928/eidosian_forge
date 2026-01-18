from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def get_channel_direct_dict(self):
    """ get channel direct attributes dict."""
    channel_direct_info = dict()
    conf_str = CE_NC_GET_CHANNEL_DIRECT_INFO % self.channel_out_direct
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return channel_direct_info
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    channel_direct_info['channelDirectInfos'] = list()
    dir_channels = root.findall('syslog/icDirChannels/icDirChannel')
    if dir_channels:
        for ic_dir_channel in dir_channels:
            channel_direct_dict = dict()
            for ele in ic_dir_channel:
                if ele.tag in ['icOutDirect', 'icCfgChnlId']:
                    channel_direct_dict[ele.tag] = ele.text
            channel_direct_info['channelDirectInfos'].append(channel_direct_dict)
    return channel_direct_info