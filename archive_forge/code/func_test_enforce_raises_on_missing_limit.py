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
def test_enforce_raises_on_missing_limit(self, mock_get_reglimit, mock_get_limit):

    def mock_usage(*a):
        return {'a': 1, 'b': 1}
    project_id = uuid.uuid4().hex
    deltas = {'a': 0, 'b': 0}
    mock_get_reglimit.return_value = None
    mock_get_limit.return_value = None
    enforcer = limit._FlatEnforcer(mock_usage)
    self.assertRaises(exception.ProjectOverLimit, enforcer.enforce, project_id, deltas)
    self.assertRaises(exception.ProjectOverLimit, enforcer.enforce, None, deltas)