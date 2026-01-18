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
def test_deltas_must_be_a_dictionary(self):
    project_id = uuid.uuid4().hex
    invalid_delta_types = [uuid.uuid4().hex, 5, 5.1, True, [], None, {}]
    enforcer = limit.Enforcer(self._get_usage_for_project)
    for invalid_delta in invalid_delta_types:
        self.assertRaises(ValueError, enforcer.enforce, project_id, invalid_delta)