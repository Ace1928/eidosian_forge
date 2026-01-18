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
def test_create_err_rpcerr(self):
    stack_name = 'wordpress'
    json_template = json.dumps(self.template)
    params = {'Action': 'CreateStack', 'StackName': stack_name, 'TemplateBody': '%s' % json_template, 'TimeoutInMinutes': 30, 'Parameters.member.1.ParameterKey': 'InstanceType', 'Parameters.member.1.ParameterValue': 'm1.xlarge'}
    engine_parms = {u'InstanceType': u'm1.xlarge'}
    engine_args = {'timeout_mins': u'30'}
    dummy_req = self._dummy_GET_request(params)
    m_f = self._stub_rpc_create_stack_call_failure(dummy_req.context, stack_name, engine_parms, engine_args, AttributeError(), direct_mock=False)
    failure = heat_exception.UnknownUserParameter(key='test')
    m_f2 = self._stub_rpc_create_stack_call_failure(dummy_req.context, stack_name, engine_parms, engine_args, failure, False, direct_mock=False)
    failure = heat_exception.UserParameterMissing(key='test')
    m_f3 = self._stub_rpc_create_stack_call_failure(dummy_req.context, stack_name, engine_parms, engine_args, failure, False, direct_mock=False)
    self.m_call.side_effect = [m_f, m_f2, m_f3]
    result = self.controller.create(dummy_req)
    self.assertIsInstance(result, exception.HeatInvalidParameterValueError)
    result = self.controller.create(dummy_req)
    self.assertIsInstance(result, exception.HeatInvalidParameterValueError)
    result = self.controller.create(dummy_req)
    self.assertIsInstance(result, exception.HeatInvalidParameterValueError)