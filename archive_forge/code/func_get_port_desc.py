import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_port_desc(dp, waiters):
    stats = dp.ofproto_parser.OFPFeaturesRequest(dp)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    descs = []
    for msg in msgs:
        stats = msg.ports
        for stat in stats.values():
            d = {'port_no': UTIL.ofp_port_to_user(stat.port_no), 'hw_addr': stat.hw_addr, 'name': stat.name.decode('utf-8'), 'config': stat.config, 'state': stat.state, 'curr': stat.curr, 'advertised': stat.advertised, 'supported': stat.supported, 'peer': stat.peer, 'curr_speed': stat.curr_speed, 'max_speed': stat.max_speed}
            descs.append(d)
    return {str(dp.id): descs}