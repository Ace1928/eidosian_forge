from __future__ import absolute_import, division, print_function
from re import match
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
def strip_interface_speed(speed):
    """Converts symbol interface speeds to a more common notation. Example: 'speed10gig' -> '10g'"""
    if isinstance(speed, list):
        result = [match('speed[0-9]{1,3}[gm]', sp) for sp in speed]
        result = [sp.group().replace('speed', '') if result else 'unknown' for sp in result if sp]
        result = ['auto' if match('auto', sp) else sp for sp in result]
    else:
        result = match('speed[0-9]{1,3}[gm]', speed)
        result = result.group().replace('speed', '') if result else 'unknown'
        result = 'auto' if match('auto', result.lower()) else result
    return result