import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_failover_replication_group(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.38'))
    expected = {'failover_replication': {'allow_attached_volume': False, 'secondary_backend_id': None}}
    g0 = cs.groups.list()[0]
    grp = g0.failover_replication()
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)
    grp = cs.groups.failover_replication('1234')
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)
    grp = cs.groups.failover_replication(g0)
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)