import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_external_ip_address_exhausted(self):
    self._check_nexc(ne.ExternalIpAddressExhausted, _('Unable to find any IP address on external network darpanet.'), net_id='darpanet')