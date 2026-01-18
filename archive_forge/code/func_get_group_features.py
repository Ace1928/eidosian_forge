import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_group_features(dp, waiters):
    ofp = dp.ofproto
    type_convert = {ofp.OFPGT_ALL: 'ALL', ofp.OFPGT_SELECT: 'SELECT', ofp.OFPGT_INDIRECT: 'INDIRECT', ofp.OFPGT_FF: 'FF'}
    cap_convert = {ofp.OFPGFC_SELECT_WEIGHT: 'SELECT_WEIGHT', ofp.OFPGFC_SELECT_LIVENESS: 'SELECT_LIVENESS', ofp.OFPGFC_CHAINING: 'CHAINING', ofp.OFPGFC_CHAINING_CHECKS: 'CHAINING_CHECKS'}
    act_convert = {ofp.OFPAT_OUTPUT: 'OUTPUT', ofp.OFPAT_COPY_TTL_OUT: 'COPY_TTL_OUT', ofp.OFPAT_COPY_TTL_IN: 'COPY_TTL_IN', ofp.OFPAT_SET_MPLS_TTL: 'SET_MPLS_TTL', ofp.OFPAT_DEC_MPLS_TTL: 'DEC_MPLS_TTL', ofp.OFPAT_PUSH_VLAN: 'PUSH_VLAN', ofp.OFPAT_POP_VLAN: 'POP_VLAN', ofp.OFPAT_PUSH_MPLS: 'PUSH_MPLS', ofp.OFPAT_POP_MPLS: 'POP_MPLS', ofp.OFPAT_SET_QUEUE: 'SET_QUEUE', ofp.OFPAT_GROUP: 'GROUP', ofp.OFPAT_SET_NW_TTL: 'SET_NW_TTL', ofp.OFPAT_DEC_NW_TTL: 'DEC_NW_TTL', ofp.OFPAT_SET_FIELD: 'SET_FIELD'}
    stats = dp.ofproto_parser.OFPGroupFeaturesStatsRequest(dp, 0)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    features = []
    for msg in msgs:
        feature = msg.body
        types = []
        for k, v in type_convert.items():
            if 1 << k & feature.types:
                types.append(v)
        capabilities = []
        for k, v in cap_convert.items():
            if k & feature.capabilities:
                capabilities.append(v)
        max_groups = []
        for k, v in type_convert.items():
            max_groups.append({v: feature.max_groups[k]})
        actions = []
        for k1, v1 in type_convert.items():
            acts = []
            for k2, v2 in act_convert.items():
                if 1 << k2 & feature.actions[k1]:
                    acts.append(v2)
            actions.append({v1: acts})
        f = {'types': types, 'capabilities': capabilities, 'max_groups': max_groups, 'actions': actions}
        features.append(f)
    return {str(dp.id): features}