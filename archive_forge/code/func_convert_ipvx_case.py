from osc_lib import exceptions
from openstackclient.i18n import _
def convert_ipvx_case(string):
    if string.lower() == 'ipv4':
        return 'IPv4'
    if string.lower() == 'ipv6':
        return 'IPv6'
    return string