import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_replication_targets(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.38'))
    expected = {'list_replication_targets': {}}
    g0 = cs.groups.list()[0]
    grp = g0.list_replication_targets()
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)
    grp = cs.groups.list_replication_targets('1234')
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)
    grp = cs.groups.list_replication_targets(g0)
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)