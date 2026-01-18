import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.lib import ofctl_utils
def mod_meter_entry(dp, meter, cmd):
    flags = 0
    if 'flags' in meter:
        meter_flags = meter['flags']
        if not isinstance(meter_flags, list):
            meter_flags = [meter_flags]
        for flag in meter_flags:
            t = UTIL.ofp_meter_flags_from_user(flag)
            f = t if t != flag else None
            if f is None:
                LOG.error('Unknown meter flag: %s', flag)
                continue
            flags |= f
    meter_id = UTIL.ofp_meter_from_user(meter.get('meter_id', 0))
    bands = []
    for band in meter.get('bands', []):
        band_type = band.get('type')
        rate = str_to_int(band.get('rate', 0))
        burst_size = str_to_int(band.get('burst_size', 0))
        if band_type == 'DROP':
            bands.append(dp.ofproto_parser.OFPMeterBandDrop(rate, burst_size))
        elif band_type == 'DSCP_REMARK':
            prec_level = str_to_int(band.get('prec_level', 0))
            bands.append(dp.ofproto_parser.OFPMeterBandDscpRemark(rate, burst_size, prec_level))
        elif band_type == 'EXPERIMENTER':
            experimenter = str_to_int(band.get('experimenter', 0))
            bands.append(dp.ofproto_parser.OFPMeterBandExperimenter(rate, burst_size, experimenter))
        else:
            LOG.error('Unknown band type: %s', band_type)
    meter_mod = dp.ofproto_parser.OFPMeterMod(dp, cmd, flags, meter_id, bands)
    ofctl_utils.send_msg(dp, meter_mod, LOG)