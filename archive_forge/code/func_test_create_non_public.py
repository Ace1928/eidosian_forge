from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import group_types
def test_create_non_public(self):
    t = cs.group_types.create('test-type-3', 'test-type-3-desc', False)
    cs.assert_called('POST', '/group_types', {'group_type': {'name': 'test-type-3', 'description': 'test-type-3-desc', 'is_public': False}})
    self.assertIsInstance(t, group_types.GroupType)
    self._assert_request_id(t)