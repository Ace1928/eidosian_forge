import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_data_optional_key_invalid(self):
    data = [{'opt_name': 'a', 'opt_value': 'A'}, {'opt_name': 'b', 'opt_value': 'B', 'ip_version': '3'}]
    self.assertRaisesRegex(n_exc.InvalidInput, 'No valid key specs', validators.validate_any_key_specs_or_none, data, key_specs=extra_dhcp_opt.EXTRA_DHCP_OPT_KEY_SPECS)