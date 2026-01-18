import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def generate_default_ethertype(protocol):
    if protocol == 'icmpv6':
        return 'IPv6'
    return 'IPv4'