import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def parse_rp_pp_without_direction(pkt_rates, host):
    """Parse: resource_provider_packet_processing_without_direction.

    Input in the config:
        resource_provider_packet_processing_without_direction =
            host0:10000,host1:,host2,:0
    Input here:
        ['host0:10000', 'host1:', 'host2', ':0']
    Output:
        {
            'host0': {'any': 10000},
            'host1': {'any': None},
            'host2': {'any': None},
            '<DEFAULT.host>': {'any': 0},
        }

    :param pkt_rates: The list of 'hypervisor:pkt_rate' config options
                      as pre-parsed by oslo_config.
    :param host: Hostname that will be used as a default key value if the user
                 did not provide hypervisor name.
    :raises: ValueError on invalid input.
    :returns: The fully parsed pkt rate config as a dict of dicts.
    """
    try:
        cfg = _parse_rp_options(pkt_rates, (const.ANY_DIRECTION,))
        _rp_pp_set_default_hypervisor(cfg, host)
    except ValueError as e:
        raise ValueError(_('Cannot parse resource_provider_packet_processing_without_direction. %s') % e) from e
    return cfg