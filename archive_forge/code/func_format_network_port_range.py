from osc_lib import exceptions
from openstackclient.i18n import _
def format_network_port_range(rule):
    port_range = ''
    if is_icmp_protocol(rule['protocol']):
        if rule['port_range_min']:
            port_range += 'type=' + str(rule['port_range_min'])
        if rule['port_range_max']:
            port_range += ':code=' + str(rule['port_range_max'])
    elif rule['port_range_min'] or rule['port_range_max']:
        port_range_min = str(rule['port_range_min'])
        port_range_max = str(rule['port_range_max'])
        if rule['port_range_min'] is None:
            port_range_min = port_range_max
        if rule['port_range_max'] is None:
            port_range_max = port_range_min
        port_range = port_range_min + ':' + port_range_max
    return port_range