from unittest import mock
from novaclient import api_versions
from novaclient import exceptions as exc
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import versions
def test_get_current(self):
    self.cs.callback = []
    v = self.cs.versions.get_current()
    self.assert_request_id(v, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', 'http://nova-api:8774/v2.1/')