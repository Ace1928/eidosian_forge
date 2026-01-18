import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_port_stats(dp, waiters, port=None):
    if port is None:
        port = dp.ofproto.OFPP_ANY
    else:
        port = str_to_int(port)
    stats = dp.ofproto_parser.OFPPortStatsRequest(dp, port, 0)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    ports = []
    for msg in msgs:
        for stats in msg.body:
            s = {'port_no': UTIL.ofp_port_to_user(stats.port_no), 'rx_packets': stats.rx_packets, 'tx_packets': stats.tx_packets, 'rx_bytes': stats.rx_bytes, 'tx_bytes': stats.tx_bytes, 'rx_dropped': stats.rx_dropped, 'tx_dropped': stats.tx_dropped, 'rx_errors': stats.rx_errors, 'tx_errors': stats.tx_errors, 'rx_frame_err': stats.rx_frame_err, 'rx_over_err': stats.rx_over_err, 'rx_crc_err': stats.rx_crc_err, 'collisions': stats.collisions}
            ports.append(s)
    return {str(dp.id): ports}