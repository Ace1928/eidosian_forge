from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_consistencygroup(self):
    lst = cs.consistencygroups.list()
    cs.assert_called('GET', '/consistencygroups/detail')
    self._assert_request_id(lst)