import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.lib import ofctl_utils
def get_queue_desc(dp, waiters, port_no=None, queue_id=None, to_user=True):
    if port_no is None:
        port_no = dp.ofproto.OFPP_ANY
    else:
        port_no = UTIL.ofp_port_from_user(port_no)
    if queue_id is None:
        queue_id = dp.ofproto.OFPQ_ALL
    else:
        queue_id = UTIL.ofp_queue_from_user(queue_id)
    stats = dp.ofproto_parser.OFPQueueDescStatsRequest(dp, 0, port_no, queue_id)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    configs = []
    for msg in msgs:
        for queue in msg.body:
            q = queue.to_jsondict()[queue.__class__.__name__]
            prop_list = []
            for prop in queue.properties:
                p = prop.to_jsondict()[prop.__class__.__name__]
                if to_user:
                    t = UTIL.ofp_queue_desc_prop_type_to_user(prop.type)
                    p['type'] = t if t != prop.type else 'UNKNOWN'
                prop_list.append(p)
            q['properties'] = prop_list
            configs.append(q)
    return wrap_dpid_dict(dp, configs, to_user)