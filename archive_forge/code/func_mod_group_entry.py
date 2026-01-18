import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def mod_group_entry(dp, group, cmd):
    type_convert = {'ALL': dp.ofproto.OFPGT_ALL, 'SELECT': dp.ofproto.OFPGT_SELECT, 'INDIRECT': dp.ofproto.OFPGT_INDIRECT, 'FF': dp.ofproto.OFPGT_FF}
    type_ = type_convert.get(group.get('type', 'ALL'))
    if type_ is None:
        LOG.error('Unknown group type: %s', group.get('type'))
    group_id = UTIL.ofp_group_from_user(group.get('group_id', 0))
    buckets = []
    for bucket in group.get('buckets', []):
        weight = str_to_int(bucket.get('weight', 0))
        watch_port = str_to_int(bucket.get('watch_port', dp.ofproto.OFPP_ANY))
        watch_group = str_to_int(bucket.get('watch_group', dp.ofproto.OFPG_ANY))
        actions = []
        for dic in bucket.get('actions', []):
            action = to_action(dp, dic)
            if action is not None:
                actions.append(action)
        buckets.append(dp.ofproto_parser.OFPBucket(weight, watch_port, watch_group, actions))
    group_mod = dp.ofproto_parser.OFPGroupMod(dp, cmd, type_, group_id, buckets)
    ofctl_utils.send_msg(dp, group_mod, LOG)