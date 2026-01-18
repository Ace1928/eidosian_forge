import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_group_stats(dp, waiters, group_id=None):
    if group_id is None:
        group_id = dp.ofproto.OFPG_ALL
    else:
        group_id = str_to_int(group_id)
    stats = dp.ofproto_parser.OFPGroupStatsRequest(dp, group_id, 0)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    groups = []
    for msg in msgs:
        for stats in msg.body:
            bucket_counters = []
            for bucket_counter in stats.bucket_counters:
                c = {'packet_count': bucket_counter.packet_count, 'byte_count': bucket_counter.byte_count}
                bucket_counters.append(c)
            g = {'length': stats.length, 'group_id': UTIL.ofp_group_to_user(stats.group_id), 'ref_count': stats.ref_count, 'packet_count': stats.packet_count, 'byte_count': stats.byte_count, 'bucket_stats': bucket_counters}
            groups.append(g)
    return {str(dp.id): groups}