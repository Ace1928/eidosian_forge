import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def parse_rp_pp_with_direction(pkt_rates, host):
    """Parse and validate: resource_provider_packet_processing_with_direction.

    Input in the config:
        resource_provider_packet_processing_with_direction =
            host0:10000:10000,host1::10000,host2::,host3,:0:0
    Input here:
        ['host0:10000:10000', 'host1::10000', 'host2::', 'host3', ':0:0']
    Output:
        {
            'host0': {'egress': 10000, 'ingress': 10000},
            'host1': {'egress': None, 'ingress': 10000},
            'host2': {'egress': None, 'ingress': None},
            'host3': {'egress': None, 'ingress': None},
            '<host>': {'egress': 0, 'ingress': 0},
        }

    :param pkt_rates: The list of 'hypervisor:egress:ingress' pkt rate
                       config options as pre-parsed by oslo_config.
    :param host: Hostname that will be used as a default key value if the user
                 did not provide hypervisor name.
    :raises: ValueError on invalid input.
    :returns: The fully parsed pkt rate config as a dict of dicts.
    """
    try:
        cfg = _parse_rp_options(pkt_rates, (const.EGRESS_DIRECTION, const.INGRESS_DIRECTION))
        _rp_pp_set_default_hypervisor(cfg, host)
    except ValueError as e:
        raise ValueError(_('Cannot parse resource_provider_packet_processing_with_direction. %s') % e) from e
    return cfg