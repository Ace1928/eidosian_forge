from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def get_multicast_global(self):
    """get one data"""
    self.multicast_global_info['multicast_global'] = list()
    getxmlstr = CE_NC_GET_MULTICAST_GLOBAL % (self.version, self.vrf)
    xml_str = get_nc_config(self.module, getxmlstr)
    if 'data/' in xml_str:
        return
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    mcast_enable = root.findall('mcastbase/mcastAfsEnables/mcastAfsEnable')
    if mcast_enable:
        for mcast_enable_key in mcast_enable:
            mcast_info = dict()
            for ele in mcast_enable_key:
                if ele.tag in ['vrfName', 'addressFamily']:
                    mcast_info[ele.tag] = ele.text
        self.multicast_global_info['multicast_global'].append(mcast_info)