import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def parse_rp_bandwidths(bandwidths):
    """Parse and validate config option: resource_provider_bandwidths.

    Input in the config:
        resource_provider_bandwidths = eth0:10000:10000,eth1::10000,eth2::,eth3
    Input here:
        ['eth0:10000:10000', 'eth1::10000', 'eth2::', 'eth3']
    Output:
        {
            'eth0': {'egress': 10000, 'ingress': 10000},
            'eth1': {'egress': None, 'ingress': 10000},
            'eth2': {'egress': None, 'ingress': None},
            'eth3': {'egress': None, 'ingress': None},
        }

    :param bandwidths: The list of 'interface:egress:ingress' bandwidth
                       config options as pre-parsed by oslo_config.
    :raises: ValueError on invalid input.
    :returns: The fully parsed bandwidth config as a dict of dicts.
    """
    try:
        return _parse_rp_options(bandwidths, (const.EGRESS_DIRECTION, const.INGRESS_DIRECTION))
    except ValueError as e:
        raise ValueError(_('Cannot parse resource_provider_bandwidths. %s') % e) from e