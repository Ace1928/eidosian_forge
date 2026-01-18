import collections
import copy
import datetime
import json
import logging
import time
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from heat.common import context
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.db import api as db_api
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_get_stack_abandon_data(self):
    tpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'param1': {'Type': 'String'}}, 'Resources': {'A': {'Type': 'GenericResourceType'}, 'B': {'Type': 'GenericResourceType'}}}
    resources = '{"A": {"status": "COMPLETE", "name": "A",\n        "resource_data": {}, "resource_id": null, "action": "INIT",\n        "type": "GenericResourceType", "metadata": {}},\n        "B": {"status": "COMPLETE", "name": "B", "resource_data": {},\n        "resource_id": null, "action": "INIT", "type": "GenericResourceType",\n        "metadata": {}}}'
    env = environment.Environment({'parameters': {'param1': 'test'}})
    self.ctx.tenant_id = '123'
    self.stack = stack.Stack(self.ctx, 'stack_details_test', template.Template(tpl, env=env), tenant_id=self.ctx.tenant_id, stack_user_project_id='234', tags=['tag1', 'tag2'])
    self.stack.store()
    info = self.stack.prepare_abandon()
    self.assertEqual('CREATE', info['action'])
    self.assertIn('id', info)
    self.assertEqual('stack_details_test', info['name'])
    self.assertEqual(json.loads(resources), info['resources'])
    self.assertEqual('IN_PROGRESS', info['status'])
    self.assertEqual(tpl, info['template'])
    self.assertEqual('123', info['project_id'])
    self.assertEqual('234', info['stack_user_project_id'])
    self.assertEqual(env.params, info['environment']['parameters'])
    self.assertEqual(['tag1', 'tag2'], info['tags'])