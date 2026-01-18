import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_table_stats(dp, waiters):
    stats = dp.ofproto_parser.OFPTableStatsRequest(dp)
    ofp = dp.ofproto
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    oxm_type_convert = {ofp.OFPXMT_OFB_IN_PORT: 'IN_PORT', ofp.OFPXMT_OFB_IN_PHY_PORT: 'IN_PHY_PORT', ofp.OFPXMT_OFB_METADATA: 'METADATA', ofp.OFPXMT_OFB_ETH_DST: 'ETH_DST', ofp.OFPXMT_OFB_ETH_SRC: 'ETH_SRC', ofp.OFPXMT_OFB_ETH_TYPE: 'ETH_TYPE', ofp.OFPXMT_OFB_VLAN_VID: 'VLAN_VID', ofp.OFPXMT_OFB_VLAN_PCP: 'VLAN_PCP', ofp.OFPXMT_OFB_IP_DSCP: 'IP_DSCP', ofp.OFPXMT_OFB_IP_ECN: 'IP_ECN', ofp.OFPXMT_OFB_IP_PROTO: 'IP_PROTO', ofp.OFPXMT_OFB_IPV4_SRC: 'IPV4_SRC', ofp.OFPXMT_OFB_IPV4_DST: 'IPV4_DST', ofp.OFPXMT_OFB_TCP_SRC: 'TCP_SRC', ofp.OFPXMT_OFB_TCP_DST: 'TCP_DST', ofp.OFPXMT_OFB_UDP_SRC: 'UDP_SRC', ofp.OFPXMT_OFB_UDP_DST: 'UDP_DST', ofp.OFPXMT_OFB_SCTP_SRC: 'SCTP_SRC', ofp.OFPXMT_OFB_SCTP_DST: 'SCTP_DST', ofp.OFPXMT_OFB_ICMPV4_TYPE: 'ICMPV4_TYPE', ofp.OFPXMT_OFB_ICMPV4_CODE: 'ICMPV4_CODE', ofp.OFPXMT_OFB_ARP_OP: 'ARP_OP', ofp.OFPXMT_OFB_ARP_SPA: 'ARP_SPA', ofp.OFPXMT_OFB_ARP_TPA: 'ARP_TPA', ofp.OFPXMT_OFB_ARP_SHA: 'ARP_SHA', ofp.OFPXMT_OFB_ARP_THA: 'ARP_THA', ofp.OFPXMT_OFB_IPV6_SRC: 'IPV6_SRC', ofp.OFPXMT_OFB_IPV6_DST: 'IPV6_DST', ofp.OFPXMT_OFB_IPV6_FLABEL: 'IPV6_FLABEL', ofp.OFPXMT_OFB_ICMPV6_TYPE: 'ICMPV6_TYPE', ofp.OFPXMT_OFB_ICMPV6_CODE: 'ICMPV6_CODE', ofp.OFPXMT_OFB_IPV6_ND_TARGET: 'IPV6_ND_TARGET', ofp.OFPXMT_OFB_IPV6_ND_SLL: 'IPV6_ND_SLL', ofp.OFPXMT_OFB_IPV6_ND_TLL: 'IPV6_ND_TLL', ofp.OFPXMT_OFB_MPLS_LABEL: 'MPLS_LABEL', ofp.OFPXMT_OFB_MPLS_TC: 'MPLS_TC'}
    act_convert = {ofp.OFPAT_OUTPUT: 'OUTPUT', ofp.OFPAT_COPY_TTL_OUT: 'COPY_TTL_OUT', ofp.OFPAT_COPY_TTL_IN: 'COPY_TTL_IN', ofp.OFPAT_SET_MPLS_TTL: 'SET_MPLS_TTL', ofp.OFPAT_DEC_MPLS_TTL: 'DEC_MPLS_TTL', ofp.OFPAT_PUSH_VLAN: 'PUSH_VLAN', ofp.OFPAT_POP_VLAN: 'POP_VLAN', ofp.OFPAT_PUSH_MPLS: 'PUSH_MPLS', ofp.OFPAT_POP_MPLS: 'POP_MPLS', ofp.OFPAT_SET_QUEUE: 'SET_QUEUE', ofp.OFPAT_GROUP: 'GROUP', ofp.OFPAT_SET_NW_TTL: 'SET_NW_TTL', ofp.OFPAT_DEC_NW_TTL: 'DEC_NW_TTL', ofp.OFPAT_SET_FIELD: 'SET_FIELD'}
    inst_convert = {ofp.OFPIT_GOTO_TABLE: 'GOTO_TABLE', ofp.OFPIT_WRITE_METADATA: 'WRITE_METADATA', ofp.OFPIT_WRITE_ACTIONS: 'WRITE_ACTIONS', ofp.OFPIT_APPLY_ACTIONS: 'APPLY_ACTIONS', ofp.OFPIT_CLEAR_ACTIONS: 'CLEAR_ACTIONS', ofp.OFPIT_EXPERIMENTER: 'EXPERIMENTER'}
    table_conf_convert = {ofp.OFPTC_TABLE_MISS_CONTROLLER: 'TABLE_MISS_CONTROLLER', ofp.OFPTC_TABLE_MISS_CONTINUE: 'TABLE_MISS_CONTINUE', ofp.OFPTC_TABLE_MISS_DROP: 'TABLE_MISS_DROP', ofp.OFPTC_TABLE_MISS_MASK: 'TABLE_MISS_MASK'}
    tables = []
    for msg in msgs:
        stats = msg.body
        for stat in stats:
            match = []
            wildcards = []
            write_setfields = []
            apply_setfields = []
            for k, v in oxm_type_convert.items():
                if 1 << k & stat.match:
                    match.append(v)
                if 1 << k & stat.wildcards:
                    wildcards.append(v)
                if 1 << k & stat.write_setfields:
                    write_setfields.append(v)
                if 1 << k & stat.apply_setfields:
                    apply_setfields.append(v)
            write_actions = []
            apply_actions = []
            for k, v in act_convert.items():
                if 1 << k & stat.write_actions:
                    write_actions.append(v)
                if 1 << k & stat.apply_actions:
                    apply_actions.append(v)
            instructions = []
            for k, v in inst_convert.items():
                if 1 << k & stat.instructions:
                    instructions.append(v)
            config = []
            for k, v in table_conf_convert.items():
                if 1 << k & stat.config:
                    config.append(v)
            s = {'table_id': UTIL.ofp_table_to_user(stat.table_id), 'name': stat.name.decode('utf-8'), 'match': match, 'wildcards': wildcards, 'write_actions': write_actions, 'apply_actions': apply_actions, 'write_setfields': write_setfields, 'apply_setfields': apply_setfields, 'metadata_match': stat.metadata_match, 'metadata_write': stat.metadata_write, 'instructions': instructions, 'config': config, 'max_entries': stat.max_entries, 'active_count': stat.active_count, 'lookup_count': stat.lookup_count, 'matched_count': stat.matched_count}
            tables.append(s)
    return {str(dp.id): tables}