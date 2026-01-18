import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_queue_config(dp, waiters, port=None):
    ofp = dp.ofproto
    if port is None:
        port = ofp.OFPP_ANY
    else:
        port = UTIL.ofp_port_from_user(str_to_int(port))
    stats = dp.ofproto_parser.OFPQueueGetConfigRequest(dp, port)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    prop_type = {dp.ofproto.OFPQT_MIN_RATE: 'MIN_RATE', dp.ofproto.OFPQT_MAX_RATE: 'MAX_RATE', dp.ofproto.OFPQT_EXPERIMENTER: 'EXPERIMENTER'}
    configs = []
    for config in msgs:
        queue_list = []
        for queue in config.queues:
            prop_list = []
            for prop in queue.properties:
                p = {'property': prop_type.get(prop.property, 'UNKNOWN')}
                if prop.property == dp.ofproto.OFPQT_MIN_RATE or prop.property == dp.ofproto.OFPQT_MAX_RATE:
                    p['rate'] = prop.rate
                elif prop.property == dp.ofproto.OFPQT_EXPERIMENTER:
                    p['experimenter'] = prop.experimenter
                    p['data'] = prop.data
                prop_list.append(p)
            q = {'port': UTIL.ofp_port_to_user(queue.port), 'properties': prop_list, 'queue_id': UTIL.ofp_queue_to_user(queue.queue_id)}
            queue_list.append(q)
        c = {'port': UTIL.ofp_port_to_user(config.port), 'queues': queue_list}
        configs.append(c)
    return {str(dp.id): configs}