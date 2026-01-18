import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_no_network_available(self):
    self._check_nexc(ne.NoNetworkAvailable, _('Unable to create the network. No tenant network is available for allocation.'))