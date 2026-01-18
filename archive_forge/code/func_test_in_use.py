import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_in_use(self):
    self._check_nexc(ne.InUse, _('The resource is in use.'))