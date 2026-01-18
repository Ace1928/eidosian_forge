import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def match_to_str(ofmatch):
    keys = {'eth_src': 'dl_src', 'eth_dst': 'dl_dst', 'eth_type': 'dl_type', 'vlan_vid': 'dl_vlan', 'ipv4_src': 'nw_src', 'ipv4_dst': 'nw_dst', 'ip_proto': 'nw_proto', 'tcp_src': 'tp_src', 'tcp_dst': 'tp_dst', 'udp_src': 'tp_src', 'udp_dst': 'tp_dst'}
    match = {}
    ofmatch = ofmatch.to_jsondict()['OFPMatch']
    ofmatch = ofmatch['oxm_fields']
    for match_field in ofmatch:
        key = match_field['OXMTlv']['field']
        if key in keys:
            key = keys[key]
        mask = match_field['OXMTlv']['mask']
        value = match_field['OXMTlv']['value']
        if key == 'dl_vlan':
            value = match_vid_to_str(value, mask)
        elif key == 'in_port':
            value = UTIL.ofp_port_to_user(value)
        elif mask is not None:
            value = str(value) + '/' + str(mask)
        match.setdefault(key, value)
    return match