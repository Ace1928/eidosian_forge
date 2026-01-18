import json
import os
from unittest import mock
from oslo_config import fixture as config_fixture
from heat.api.aws import exception
import heat.api.cfn.v1.stacks as stacks
from heat.common import exception as heat_exception
from heat.common import identifier
from heat.common import policy
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_enforce_ise(self):
    params = {'Action': 'ListStacks'}
    dummy_req = self._dummy_GET_request(params)
    dummy_req.context.roles = ['heat_stack_user']
    mock_enforce = self.patchobject(policy.Enforcer, 'enforce')
    mock_enforce.side_effect = AttributeError
    self.assertRaises(exception.HeatInternalFailureError, self.controller._enforce, dummy_req, 'ListStacks')