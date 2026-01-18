import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_queue_stats(dp, waiters, port=None, queue_id=None):
    ofp = dp.ofproto
    if port is None:
        port = ofp.OFPP_ANY
    else:
        port = str_to_int(port)
    if queue_id is None:
        queue_id = ofp.OFPQ_ALL
    else:
        queue_id = str_to_int(queue_id)
    stats = dp.ofproto_parser.OFPQueueStatsRequest(dp, port, queue_id, 0)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    s = []
    for msg in msgs:
        stats = msg.body
        for stat in stats:
            s.append({'port_no': stat.port_no, 'queue_id': stat.queue_id, 'tx_bytes': stat.tx_bytes, 'tx_errors': stat.tx_errors, 'tx_packets': stat.tx_packets})
    return {str(dp.id): s}