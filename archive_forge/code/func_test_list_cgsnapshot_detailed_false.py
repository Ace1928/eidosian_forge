from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_cgsnapshot_detailed_false(self):
    lst = cs.cgsnapshots.list(detailed=False)
    cs.assert_called('GET', '/cgsnapshots')
    self._assert_request_id(lst)