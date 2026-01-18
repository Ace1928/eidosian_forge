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
@mock.patch.object(limit._EnforcerUtils, '_get_project_limit')
@mock.patch.object(limit._EnforcerUtils, '_get_registered_limit')
def test_calculate_and_enforce_some_missing(self, mock_get_reglimit, mock_get_limit):
    reg_limits = {'a': mock.MagicMock(default_limit=10), 'b': mock.MagicMock(default_limit=10)}
    prj_limits = {('bar', 'b'): mock.MagicMock(resource_limit=6)}
    mock_get_reglimit.side_effect = lambda r: reg_limits.get(r)
    mock_get_limit.side_effect = lambda p, r: prj_limits.get((p, r))
    mock_usage = mock.MagicMock()
    mock_usage.return_value = {'a': 5, 'b': 5, 'c': 5}
    enforcer = limit.Enforcer(mock_usage)
    expected = {'a': limit.ProjectUsage(10, 5), 'b': limit.ProjectUsage(6, 5), 'c': limit.ProjectUsage(0, 5)}
    self.assertEqual(expected, enforcer.calculate_usage('bar', ['a', 'b', 'c']))
    self.assertRaises(exception.ProjectOverLimit, enforcer.enforce, 'bar', {'a': 1, 'b': 0, 'c': 1})