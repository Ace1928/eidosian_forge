import base64
import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
from os_ken.lib import ofctl_utils
def get_flow_desc_stats(dp, waiters, flow=None, to_user=True):
    flow = flow if flow else {}
    table_id = UTIL.ofp_table_from_user(flow.get('table_id', dp.ofproto.OFPTT_ALL))
    flags = str_to_int(flow.get('flags', 0))
    out_port = UTIL.ofp_port_from_user(flow.get('out_port', dp.ofproto.OFPP_ANY))
    out_group = UTIL.ofp_group_from_user(flow.get('out_group', dp.ofproto.OFPG_ANY))
    cookie = str_to_int(flow.get('cookie', 0))
    cookie_mask = str_to_int(flow.get('cookie_mask', 0))
    match = to_match(dp, flow.get('match', {}))
    priority = str_to_int(flow.get('priority', -1))
    stats = dp.ofproto_parser.OFPFlowDescStatsRequest(dp, flags, table_id, out_port, out_group, cookie, cookie_mask, match)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    flows = []
    for msg in msgs:
        for stats in msg.body:
            if 0 <= priority != stats.priority:
                continue
            s = stats.to_jsondict()[stats.__class__.__name__]
            s['instructions'] = instructions_to_str(stats.instructions)
            s['stats'] = stats_to_str(stats.stats)
            s['match'] = match_to_str(stats.match)
            flows.append(s)
    return wrap_dpid_dict(dp, flows, to_user)