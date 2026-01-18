import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_create_with_template_failure(self):

    class StackValidationFailed_Remote(exception.StackValidationFailed):
        pass
    child_env = {'parameter_defaults': {}, 'event_sinks': [], 'parameters': self.params, 'resource_registry': {'resources': {}}}
    self.parent_resource.child_params = mock.Mock(return_value=self.params)
    res_name = self.parent_resource.physical_resource_name()
    rpcc = mock.Mock()
    self.parent_resource.rpc_client = rpcc
    remote_exc = StackValidationFailed_Remote(message='oops')
    rpcc.return_value._create_stack.side_effect = remote_exc
    self.assertRaises(exception.ResourceFailure, self.parent_resource.create_with_template, self.empty_temp, user_params=self.params, timeout_mins=self.timeout_mins, adopt_data=self.adopt_data)
    if self.adopt_data is not None:
        adopt_data_str = json.dumps(self.adopt_data)
        tmpl_args = {'template': self.empty_temp.t, 'params': child_env, 'files': {}}
    else:
        adopt_data_str = None
        tmpl_args = {'template_id': self.IntegerMatch(), 'template': None, 'params': None, 'files': None}
    rpcc.return_value._create_stack.assert_called_once_with(self.ctx, stack_name=res_name, args={rpc_api.PARAM_DISABLE_ROLLBACK: True, rpc_api.PARAM_ADOPT_STACK_DATA: adopt_data_str, rpc_api.PARAM_TIMEOUT: self.timeout_mins}, environment_files=None, stack_user_project_id='aprojectid', parent_resource_name='test', user_creds_id='uc123', owner_id=self.parent_stack.id, nested_depth=1, **tmpl_args)
    if self.adopt_data is None:
        stored_tmpl_id = tmpl_args['template_id'].match
        self.assertIsNotNone(stored_tmpl_id)
        self.assertRaises(exception.NotFound, raw_template.RawTemplate.get_by_id, self.ctx, stored_tmpl_id)