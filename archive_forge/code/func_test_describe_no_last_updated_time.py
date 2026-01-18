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
def test_describe_no_last_updated_time(self):
    params = {'Action': 'DescribeStacks'}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'DescribeStacks')
    engine_resp = [{u'updated_time': None, u'parameters': {}, u'stack_action': u'CREATE', u'stack_status': u'COMPLETE'}]
    self.m_call.return_value = engine_resp
    response = self.controller.describe(dummy_req)
    result = response['DescribeStacksResponse']['DescribeStacksResult']
    stack = result['Stacks'][0]
    self.assertNotIn('LastUpdatedTime', stack)
    self.m_call.assert_called_once_with(dummy_req.context, ('show_stack', {'stack_identity': None, 'resolve_outputs': True}), version='1.20')