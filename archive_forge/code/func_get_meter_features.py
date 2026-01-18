import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.lib import ofctl_utils
def get_meter_features(dp, waiters, to_user=True):
    ofp = dp.ofproto
    type_convert = {ofp.OFPMBT_DROP: 'DROP', ofp.OFPMBT_DSCP_REMARK: 'DSCP_REMARK'}
    capa_convert = {ofp.OFPMF_KBPS: 'KBPS', ofp.OFPMF_PKTPS: 'PKTPS', ofp.OFPMF_BURST: 'BURST', ofp.OFPMF_STATS: 'STATS'}
    stats = dp.ofproto_parser.OFPMeterFeaturesStatsRequest(dp, 0)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    features = []
    for msg in msgs:
        for feature in msg.body:
            band_types = []
            for k, v in type_convert.items():
                if 1 << k & feature.band_types:
                    if to_user:
                        band_types.append(v)
                    else:
                        band_types.append(k)
            capabilities = []
            for k, v in sorted(capa_convert.items()):
                if k & feature.capabilities:
                    if to_user:
                        capabilities.append(v)
                    else:
                        capabilities.append(k)
            f = {'max_meter': feature.max_meter, 'band_types': band_types, 'capabilities': capabilities, 'max_bands': feature.max_bands, 'max_color': feature.max_color}
            features.append(f)
    return wrap_dpid_dict(dp, features, to_user)