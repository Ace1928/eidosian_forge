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
def test__test__validate_list_of_items_non_empty(self):
    items = None
    msg = validators._validate_list_of_items_non_empty(mock.Mock(), items)
    error = "'%s' is not a list" % items
    self.assertEqual(error, msg)
    items = []
    msg = validators._validate_list_of_items_non_empty(mock.Mock(), items)
    self.assertEqual('List should not be empty', msg)