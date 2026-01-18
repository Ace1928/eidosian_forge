import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_group_desc(dp, waiters):
    type_convert = {dp.ofproto.OFPGT_ALL: 'ALL', dp.ofproto.OFPGT_SELECT: 'SELECT', dp.ofproto.OFPGT_INDIRECT: 'INDIRECT', dp.ofproto.OFPGT_FF: 'FF'}
    stats = dp.ofproto_parser.OFPGroupDescStatsRequest(dp, 0)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    descs = []
    for msg in msgs:
        for stats in msg.body:
            buckets = []
            for bucket in stats.buckets:
                actions = []
                for action in bucket.actions:
                    actions.append(action_to_str(action))
                b = {'weight': bucket.weight, 'watch_port': bucket.watch_port, 'watch_group': bucket.watch_group, 'actions': actions}
                buckets.append(b)
            d = {'type': type_convert.get(stats.type), 'group_id': UTIL.ofp_group_to_user(stats.group_id), 'buckets': buckets}
            descs.append(d)
    return {str(dp.id): descs}