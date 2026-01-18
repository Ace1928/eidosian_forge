import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_invalid_ip_for_subnet(self):
    self._check_nexc(ne.InvalidIpForSubnet, _('IP address 300.400.500.600 is not a valid IP for the specified subnet.'), ip_address='300.400.500.600')