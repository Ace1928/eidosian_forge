from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_nve_dict(self, nve_name):
    """ get nve interface attributes dict."""
    nve_info = dict()
    conf_str = CE_NC_GET_NVE_INFO % nve_name
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return nve_info
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    nvo3 = root.find('nvo3/nvo3Nves/nvo3Nve')
    if nvo3:
        for nve in nvo3:
            if nve.tag in ['srcAddr', 'ifName', 'nveType']:
                nve_info[nve.tag] = nve.text
    nve_info['vni_peer_protocols'] = list()
    vni_members = root.findall('nvo3/nvo3Nves/nvo3Nve/vniMembers/vniMember')
    if vni_members:
        for member in vni_members:
            vni_dict = dict()
            for ele in member:
                if ele.tag in ['vniId', 'protocol']:
                    vni_dict[ele.tag] = ele.text
            nve_info['vni_peer_protocols'].append(vni_dict)
    nve_info['vni_peer_ips'] = list()
    re_find = re.findall('<vniId>(.*?)</vniId>\\s*<protocol>(.*?)</protocol>\\s*<nvo3VniPeers>(.*?)</nvo3VniPeers>', xml_str)
    if re_find:
        for vni_peers in re_find:
            vni_info = dict()
            vni_peer = re.findall('<peerAddr>(.*?)</peerAddr>', vni_peers[2])
            if vni_peer:
                vni_info['vniId'] = vni_peers[0]
                vni_peer_list = list()
                for peer in vni_peer:
                    vni_peer_list.append(peer)
                vni_info['peerAddr'] = vni_peer_list
            nve_info['vni_peer_ips'].append(vni_info)
    return nve_info