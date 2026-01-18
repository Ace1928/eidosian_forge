from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_kvp_str_to_list_succeeds_for_one_equals(self):
    result = converters.convert_kvp_str_to_list('a=')
    self.assertEqual(['a', ''], result)