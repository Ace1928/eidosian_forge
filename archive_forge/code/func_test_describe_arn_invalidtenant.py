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
def test_describe_arn_invalidtenant(self):
    stack_name = u'wordpress'
    stack_identifier = identifier.HeatIdentifier('wibble', stack_name, '6')
    identity = dict(stack_identifier)
    params = {'Action': 'DescribeStacks', 'StackName': stack_identifier.arn()}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'DescribeStacks')
    exc = heat_exception.InvalidTenant(target='test', actual='test')
    self.m_call.side_effect = exc
    result = self.controller.describe(dummy_req)
    self.assertIsInstance(result, exception.HeatInvalidParameterValueError)
    self.m_call.assert_called_once_with(dummy_req.context, ('show_stack', {'stack_identity': identity, 'resolve_outputs': True}), version='1.20')