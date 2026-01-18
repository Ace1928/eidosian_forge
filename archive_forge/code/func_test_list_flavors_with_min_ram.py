from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_list_flavors_with_min_ram(self):
    fl = self.cs.flavors.list(min_ram=512)
    self.assert_request_id(fl, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/flavors/detail?minRam=512')