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
def test_calculate_usage(self, mock_get_limits):
    mock_usage = mock.MagicMock()
    mock_usage.return_value = {'a': 1, 'b': 2}
    project_id = uuid.uuid4().hex
    mock_get_limits.return_value = [('a', 10), ('b', 5)]
    expected = {'a': limit.ProjectUsage(10, 1), 'b': limit.ProjectUsage(5, 2)}
    enforcer = limit.Enforcer(mock_usage)
    self.assertEqual(expected, enforcer.calculate_usage(project_id, ['a', 'b']))