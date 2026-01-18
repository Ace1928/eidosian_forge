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
def test_enforce(self, mock_get_limits):
    mock_usage = mock.MagicMock()
    project_id = uuid.uuid4().hex
    deltas = {'a': 1, 'b': 1}
    mock_get_limits.return_value = [('a', 1), ('b', 2)]
    mock_usage.return_value = {'a': 0, 'b': 1}
    enforcer = limit._FlatEnforcer(mock_usage)
    enforcer.enforce(project_id, deltas)
    self.mock_conn.get_endpoint.assert_called_once_with('ENDPOINT_ID')
    mock_get_limits.assert_called_once_with(project_id, ['a', 'b'])
    mock_usage.assert_called_once_with(project_id, ['a', 'b'])