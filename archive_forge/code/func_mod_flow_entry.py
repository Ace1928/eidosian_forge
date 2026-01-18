import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def mod_flow_entry(dp, flow, cmd):
    cookie = str_to_int(flow.get('cookie', 0))
    cookie_mask = str_to_int(flow.get('cookie_mask', 0))
    table_id = UTIL.ofp_table_from_user(flow.get('table_id', 0))
    idle_timeout = str_to_int(flow.get('idle_timeout', 0))
    hard_timeout = str_to_int(flow.get('hard_timeout', 0))
    priority = str_to_int(flow.get('priority', 0))
    buffer_id = UTIL.ofp_buffer_from_user(flow.get('buffer_id', dp.ofproto.OFP_NO_BUFFER))
    out_port = UTIL.ofp_port_from_user(flow.get('out_port', dp.ofproto.OFPP_ANY))
    out_group = UTIL.ofp_group_from_user(flow.get('out_group', dp.ofproto.OFPG_ANY))
    flags = str_to_int(flow.get('flags', 0))
    match = to_match(dp, flow.get('match', {}))
    inst = to_actions(dp, flow.get('actions', []))
    flow_mod = dp.ofproto_parser.OFPFlowMod(dp, cookie, cookie_mask, table_id, cmd, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags, match, inst)
    ofctl_utils.send_msg(dp, flow_mod, LOG)