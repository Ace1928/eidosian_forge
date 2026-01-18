import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_group_with_volume_types(self):
    grp = cs.groups.create('my_group_type', 'type1,type2', name='group')
    expected = {'group': {'description': None, 'availability_zone': None, 'name': 'group', 'group_type': 'my_group_type', 'volume_types': ['type1', 'type2']}}
    cs.assert_called('POST', '/groups', body=expected)
    self._assert_request_id(grp)