import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def mod_port_behavior(dp, port_config):
    port_no = UTIL.ofp_port_from_user(port_config.get('port_no', 0))
    hw_addr = str(port_config.get('hw_addr'))
    config = str_to_int(port_config.get('config', 0))
    mask = str_to_int(port_config.get('mask', 0))
    advertise = str_to_int(port_config.get('advertise'))
    port_mod = dp.ofproto_parser.OFPPortMod(dp, port_no, hw_addr, config, mask, advertise)
    ofctl_utils.send_msg(dp, port_mod, LOG)