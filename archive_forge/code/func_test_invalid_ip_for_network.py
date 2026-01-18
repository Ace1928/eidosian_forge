import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_invalid_ip_for_network(self):
    self._check_nexc(ne.InvalidIpForNetwork, _('IP address shazam! is not a valid IP for any of the subnets on the specified network.'), ip_address='shazam!')