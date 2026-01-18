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
def test_describe_stack_resources_err_inval(self):
    stack_name = 'wordpress'
    params = {'Action': 'DescribeStackResources', 'StackName': stack_name, 'PhysicalResourceId': '123456'}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'DescribeStackResources')
    ret = self.controller.describe_stack_resources(dummy_req)
    self.assertIsInstance(ret, exception.HeatInvalidParameterCombinationError)