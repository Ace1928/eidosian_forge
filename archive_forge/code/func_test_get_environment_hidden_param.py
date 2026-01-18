from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.engine.cfn import template as cfntemplate
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import template as hottemplate
from heat.engine import resource as res
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_get_environment_hidden_param(self):
    env = {'parameters': {'admin': 'testuser', 'pass': 'pa55w0rd'}, 'parameter_defaults': {'secret': 'dummy'}, 'resource_registry': {'res': 'resource.yaml'}}
    t = {'heat_template_version': '2018-08-31', 'parameters': {'admin': {'type': 'string'}, 'pass': {'type': 'string', 'hidden': True}}, 'resources': {'res1': {'type': 'res'}}}
    files = {'resource.yaml': '\n                heat_template_version: 2018-08-31\n                parameters:\n                    secret:\n                        type: string\n                        hidden: true\n            '}
    tmpl = templatem.Template(t, files=files)
    stack = parser.Stack(self.ctx, 'get_env_stack', tmpl)
    stack.store()
    mock_get_stack = self.patchobject(self.eng, '_get_stack')
    mock_get_stack.return_value = mock.MagicMock()
    mock_get_stack.return_value.raw_template.environment = env
    self.patchobject(templatem.Template, 'load', return_value=tmpl)
    found = self.eng.get_environment(self.ctx, stack.identifier())
    env['parameters']['pass'] = '******'
    env['parameter_defaults']['secret'] = '******'
    self.assertEqual(env, found)