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
@tools.stack_context('service_export_stack')
def test_export_stack(self):
    cfg.CONF.set_override('enable_stack_abandon', True)
    self.patchobject(parser.Stack, 'load', return_value=self.stack)
    expected_res = {u'WebServer': {'action': 'CREATE', 'metadata': {}, 'name': u'WebServer', 'resource_data': {}, 'resource_id': '9999', 'status': 'COMPLETE', 'type': u'AWS::EC2::Instance'}}
    self.stack.tags = ['tag1', 'tag2']
    ret = self.eng.export_stack(self.ctx, self.stack.identifier())
    self.assertEqual(11, len(ret))
    self.assertEqual('CREATE', ret['action'])
    self.assertEqual('COMPLETE', ret['status'])
    self.assertEqual('service_export_stack', ret['name'])
    self.assertEqual({}, ret['files'])
    self.assertIn('id', ret)
    self.assertEqual(expected_res, ret['resources'])
    self.assertEqual(self.stack.t.t, ret['template'])
    self.assertIn('project_id', ret)
    self.assertIn('stack_user_project_id', ret)
    self.assertIn('environment', ret)
    self.assertIn('files', ret)
    self.assertEqual(['tag1', 'tag2'], ret['tags'])