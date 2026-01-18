import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_group_from_src_snap(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.14'))
    grp = cs.groups.create_from_src('5678', None, name='group')
    expected = {'create-from-src': {'description': None, 'name': 'group', 'group_snapshot_id': '5678'}}
    cs.assert_called('POST', '/groups/action', body=expected)
    self._assert_request_id(grp)