from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_trunk_dict(self, trunk_id):
    """ get one interface attributes dict."""
    trunk_info = dict()
    conf_str = CE_NC_GET_TRUNK % trunk_id
    recv_xml = get_nc_config(self.module, conf_str)
    if '<data/>' in recv_xml:
        return trunk_info
    base = re.findall('.*<ifName>(.*)</ifName>.*\\s*<minUpNum>(.*)</minUpNum>.*\\s*<maxUpNum>(.*)</maxUpNum>.*\\s*<trunkType>(.*)</trunkType>.*\\s*<hashType>(.*)</hashType>.*\\s*<workMode>(.*)</workMode>.*\\s*<upMemberIfNum>(.*)</upMemberIfNum>.*\\s*<memberIfNum>(.*)</memberIfNum>.*', recv_xml)
    if base:
        trunk_info = dict(ifName=base[0][0], trunkId=base[0][0].lower().replace('eth-trunk', '').replace(' ', ''), minUpNum=base[0][1], maxUpNum=base[0][2], trunkType=base[0][3], hashType=base[0][4], workMode=base[0][5], upMemberIfNum=base[0][6], memberIfNum=base[0][7])
    member = re.findall('.*<memberIfName>(.*)</memberIfName>.*\\s*<memberIfState>(.*)</memberIfState>.*', recv_xml)
    trunk_info['TrunkMemberIfs'] = list()
    for mem in member:
        trunk_info['TrunkMemberIfs'].append(dict(memberIfName=mem[0], memberIfState=mem[1]))
    return trunk_info