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
def test_describe_arn(self):
    stack_name = u'wordpress'
    stack_identifier = identifier.HeatIdentifier('t', stack_name, '6')
    identity = dict(stack_identifier)
    params = {'Action': 'DescribeStacks', 'StackName': stack_identifier.arn()}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'DescribeStacks')
    engine_resp = [{u'stack_identity': {u'tenant': u't', u'stack_name': u'wordpress', u'stack_id': u'6', u'path': u''}, u'updated_time': u'2012-07-09T09:13:11Z', u'parameters': {u'DBUsername': u'admin', u'LinuxDistribution': u'F17', u'InstanceType': u'm1.large', u'DBRootPassword': u'admin', u'DBPassword': u'admin', u'DBName': u'wordpress'}, u'outputs': [{u'output_key': u'WebsiteURL', u'description': u'URL for Wordpress wiki', u'output_value': u'http://10.0.0.8/wordpress'}], u'stack_status_reason': u'Stack successfully created', u'creation_time': u'2012-07-09T09:12:45Z', u'stack_name': u'wordpress', u'notification_topics': [], u'stack_action': u'CREATE', u'stack_status': u'COMPLETE', u'description': u'blah', u'disable_rollback': 'true', u'timeout_mins': 60, u'capabilities': []}]
    self.m_call.return_value = engine_resp
    response = self.controller.describe(dummy_req)
    expected = {'DescribeStacksResponse': {'DescribeStacksResult': {'Stacks': [{'StackId': u'arn:openstack:heat::t:stacks/wordpress/6', 'StackStatusReason': u'Stack successfully created', 'Description': u'blah', 'Parameters': [{'ParameterValue': u'wordpress', 'ParameterKey': u'DBName'}, {'ParameterValue': u'admin', 'ParameterKey': u'DBPassword'}, {'ParameterValue': u'admin', 'ParameterKey': u'DBRootPassword'}, {'ParameterValue': u'admin', 'ParameterKey': u'DBUsername'}, {'ParameterValue': u'm1.large', 'ParameterKey': u'InstanceType'}, {'ParameterValue': u'F17', 'ParameterKey': u'LinuxDistribution'}], 'Outputs': [{'OutputKey': u'WebsiteURL', 'OutputValue': u'http://10.0.0.8/wordpress', 'Description': u'URL for Wordpress wiki'}], 'TimeoutInMinutes': 60, 'CreationTime': u'2012-07-09T09:12:45Z', 'Capabilities': [], 'StackName': u'wordpress', 'NotificationARNs': [], 'StackStatus': u'CREATE_COMPLETE', 'DisableRollback': 'true', 'LastUpdatedTime': u'2012-07-09T09:13:11Z'}]}}}
    stacks = response['DescribeStacksResponse']['DescribeStacksResult']['Stacks']
    stacks[0]['Parameters'] = sorted(stacks[0]['Parameters'], key=lambda k: k['ParameterKey'])
    response['DescribeStacksResponse']['DescribeStacksResult'] = {'Stacks': stacks}
    self.assertEqual(expected, response)
    self.m_call.assert_called_once_with(dummy_req.context, ('show_stack', {'stack_identity': identity, 'resolve_outputs': True}), version='1.20')