from unittest import mock
import uuid
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from openstack.identity.v3 import registered_limit
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
@mock.patch.object(limit._EnforcerUtils, 'get_project_limits')
def test_enforce_raises_on_over(self, mock_get_limits):
    mock_usage = mock.MagicMock()
    project_id = uuid.uuid4().hex
    deltas = {'a': 2, 'b': 1}
    mock_get_limits.return_value = [('a', 1), ('b', 2)]
    mock_usage.return_value = {'a': 0, 'b': 1}
    enforcer = limit._FlatEnforcer(mock_usage)
    e = self.assertRaises(exception.ProjectOverLimit, enforcer.enforce, project_id, deltas)
    expected = 'Project %s is over a limit for [Resource a is over limit of 1 due to current usage 0 and delta 2]'
    self.assertEqual(expected % project_id, str(e))
    self.assertEqual(project_id, e.project_id)
    self.assertEqual(1, len(e.over_limit_info_list))
    over_a = e.over_limit_info_list[0]
    self.assertEqual('a', over_a.resource_name)
    self.assertEqual(1, over_a.limit)
    self.assertEqual(0, over_a.current_usage)
    self.assertEqual(2, over_a.delta)