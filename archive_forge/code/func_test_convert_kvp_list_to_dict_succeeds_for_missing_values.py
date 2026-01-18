from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_kvp_list_to_dict_succeeds_for_missing_values(self):
    result = converters.convert_kvp_list_to_dict(['True'])
    self.assertEqual({}, result)