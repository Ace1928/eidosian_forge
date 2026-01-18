import base64
import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
from os_ken.lib import ofctl_utils
def get_meter_desc(dp, waiters, meter_id=None, to_user=True):
    flags = {dp.ofproto.OFPMF_KBPS: 'KBPS', dp.ofproto.OFPMF_PKTPS: 'PKTPS', dp.ofproto.OFPMF_BURST: 'BURST', dp.ofproto.OFPMF_STATS: 'STATS'}
    if meter_id is None:
        meter_id = dp.ofproto.OFPM_ALL
    else:
        meter_id = UTIL.ofp_meter_from_user(meter_id)
    stats = dp.ofproto_parser.OFPMeterDescStatsRequest(dp, 0, meter_id)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    configs = []
    for msg in msgs:
        for config in msg.body:
            c = config.to_jsondict()[config.__class__.__name__]
            bands = []
            for band in config.bands:
                b = band.to_jsondict()[band.__class__.__name__]
                if to_user:
                    t = UTIL.ofp_meter_band_type_to_user(band.type)
                    b['type'] = t if t != band.type else 'UNKNOWN'
                bands.append(b)
            c_flags = []
            for k, v in sorted(flags.items()):
                if k & config.flags:
                    if to_user:
                        c_flags.append(v)
                    else:
                        c_flags.append(k)
            c['flags'] = c_flags
            c['bands'] = bands
            if to_user:
                c['meter_id'] = UTIL.ofp_meter_to_user(config.meter_id)
            configs.append(c)
    return wrap_dpid_dict(dp, configs, to_user)