from __future__ import (absolute_import, division, print_function)
import sys
import socket
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def get_bfd_dict(self):
    """bfd config dict"""
    bfd_dict = dict()
    bfd_dict['global'] = dict()
    conf_str = CE_NC_GET_BFD % CE_NC_GET_BFD_GLB
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return bfd_dict
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    glb = root.find('bfd/bfdSchGlobal')
    if glb:
        for attr in glb:
            if attr.text is not None:
                bfd_dict['global'][attr.tag] = attr.text
    return bfd_dict