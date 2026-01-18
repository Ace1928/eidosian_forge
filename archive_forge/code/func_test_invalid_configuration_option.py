import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_invalid_configuration_option(self):
    self._check_nexc(ne.InvalidConfigurationOption, _('An invalid value was provided for which muppet: big bird.'), opt_name='which muppet', opt_value='big bird')