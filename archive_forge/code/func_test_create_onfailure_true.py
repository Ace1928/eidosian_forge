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
def test_create_onfailure_true(self):
    stack_name = 'wordpress'
    json_template = json.dumps(self.template)
    params = {'Action': 'CreateStack', 'StackName': stack_name, 'TemplateBody': '%s' % json_template, 'TimeoutInMinutes': 30, 'OnFailure': 'DO_NOTHING', 'Parameters.member.1.ParameterKey': 'InstanceType', 'Parameters.member.1.ParameterValue': 'm1.xlarge'}
    engine_parms = {u'InstanceType': u'm1.xlarge'}
    engine_args = {'timeout_mins': u'30', 'disable_rollback': 'true'}
    dummy_req = self._stub_rpc_create_stack_call_success(stack_name, engine_parms, engine_args, params)
    response = self.controller.create(dummy_req)
    expected = {'CreateStackResponse': {'CreateStackResult': {u'StackId': u'arn:openstack:heat::t:stacks/wordpress/1'}}}
    self.assertEqual(expected, response)
    self.m_call.assert_called_once_with(dummy_req.context, ('create_stack', {'stack_name': stack_name, 'template': self.template, 'params': engine_parms, 'files': {}, 'environment_files': None, 'files_container': None, 'args': engine_args, 'owner_id': None, 'nested_depth': 0, 'user_creds_id': None, 'parent_resource_name': None, 'stack_user_project_id': None, 'template_id': None}), version='1.36')