import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_group(self):
    lst = cs.groups.list()
    cs.assert_called('GET', '/groups/detail')
    self._assert_request_id(lst)