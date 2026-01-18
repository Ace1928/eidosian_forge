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
def test_create_err_inval_template(self):
    stack_name = 'wordpress'
    json_template = '!$%**_+}@~?'
    params = {'Action': 'CreateStack', 'StackName': stack_name, 'TemplateBody': '%s' % json_template}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'CreateStack')
    result = self.controller.create(dummy_req)
    self.assertIsInstance(result, exception.HeatInvalidParameterValueError)