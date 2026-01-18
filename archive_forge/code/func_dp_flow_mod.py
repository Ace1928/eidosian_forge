from os_ken.lib.packet import ethernet
from os_ken.lib.packet import vlan
from os_ken.ofproto import ether
from os_ken.topology import api as topo_api
def dp_flow_mod(dp, table, command, priority, match, instructions, out_port=None):
    ofproto = dp.ofproto
    ofproto_parser = dp.ofproto_parser
    if out_port is None:
        out_port = ofproto.OFPP_ANY
    flow_mod = ofproto_parser.OFPFlowMod(dp, 0, 0, table, command, 0, 0, priority, 4294967295, out_port, ofproto.OFPG_ANY, ofproto.OFPFF_CHECK_OVERLAP, match, instructions)
    dp.send_msg(flow_mod)