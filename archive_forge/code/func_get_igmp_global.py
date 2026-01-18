from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def get_igmp_global(self):
    """get igmp global data"""
    self.igmp_info_data['igmp_info'] = list()
    getxmlstr = CE_NC_GET_IGMP_GLOBAL % self.addr_family
    xml_str = get_nc_config(self.module, getxmlstr)
    if 'data/' in xml_str:
        return
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    igmp_enable = root.findall('l2mc/l2McSnpgEnables/l2McSnpgEnable')
    if igmp_enable:
        for igmp_enable_key in igmp_enable:
            igmp_global_info = dict()
            for ele in igmp_enable_key:
                if ele.tag in ['addrFamily']:
                    igmp_global_info[ele.tag] = ele.text
                    self.igmp_info_data['igmp_info'].append(igmp_global_info)