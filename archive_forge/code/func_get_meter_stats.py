import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.lib import ofctl_utils
def get_meter_stats(dp, waiters, meter_id=None, to_user=True):
    if meter_id is None:
        meter_id = dp.ofproto.OFPM_ALL
    else:
        meter_id = UTIL.ofp_meter_from_user(meter_id)
    stats = dp.ofproto_parser.OFPMeterStatsRequest(dp, 0, meter_id)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    meters = []
    for msg in msgs:
        for stats in msg.body:
            s = stats.to_jsondict()[stats.__class__.__name__]
            bands = []
            for band in stats.band_stats:
                b = band.to_jsondict()[band.__class__.__name__]
                bands.append(b)
            s['band_stats'] = bands
            if to_user:
                s['meter_id'] = UTIL.ofp_meter_to_user(stats.meter_id)
            meters.append(s)
    return wrap_dpid_dict(dp, meters, to_user)