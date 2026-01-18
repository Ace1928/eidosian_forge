from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_cgsnapshot_with_empty_search_opt(self):
    lst = cs.cgsnapshots.list(search_opts={'foo': 'bar', '123': None})
    cs.assert_called('GET', '/cgsnapshots/detail?foo=bar')
    self._assert_request_id(lst)