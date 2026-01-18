from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import group_types
def test_list_group_types_not_public(self):
    t1 = cs.group_types.list(is_public=None)
    cs.assert_called('GET', '/group_types?is_public=None')
    self._assert_request_id(t1)