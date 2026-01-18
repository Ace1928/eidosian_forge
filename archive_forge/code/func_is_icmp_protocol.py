from osc_lib import exceptions
from openstackclient.i18n import _
def is_icmp_protocol(protocol):
    if protocol in ['icmp', 'icmpv6', 'ipv6-icmp', '1', '58']:
        return True
    else:
        return False