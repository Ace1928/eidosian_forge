import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_ip_address_in_use(self):
    self._check_nexc(ne.IpAddressInUse, _('Unable to complete operation for network boredom. The IP address crazytown is in use.'), net_id='boredom', ip_address='crazytown')