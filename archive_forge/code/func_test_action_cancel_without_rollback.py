import json
from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.actions as actions
from heat.common import identifier
from heat.common import policy
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_action_cancel_without_rollback(self, mock_enforce):
    self._test_action_cancel_update(mock_enforce, False)