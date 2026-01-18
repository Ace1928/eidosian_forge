import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_group_with_volume(self):
    lst = cs.groups.list(list_volume=True)
    cs.assert_called('GET', '/groups/detail?list_volume=True')
    self._assert_request_id(lst)