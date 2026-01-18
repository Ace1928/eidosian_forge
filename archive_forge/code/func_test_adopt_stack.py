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
def test_adopt_stack(self):
    adopt_data = '{\n        "action": "CREATE",\n        "status": "COMPLETE",\n        "name": "my-test-stack-name",\n        "resources": {\n        "AResource": {\n        "status": "COMPLETE",\n        "name": "AResource",\n        "resource_data": {},\n        "metadata": {},\n        "resource_id": "test-res-id",\n        "action": "CREATE",\n        "type": "GenericResourceType"\n          }\n         }\n        }'
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'GenericResourceType'}}, 'Outputs': {'TestOutput': {'Value': {'Fn::GetAtt': ['AResource', 'Foo']}}}}
    self.stack = stack.Stack(utils.dummy_context(), 'test_stack', template.Template(tmpl), adopt_stack_data=json.loads(adopt_data))
    self.stack.store()
    self.stack.adopt()
    res = self.stack['AResource']
    self.assertEqual(u'test-res-id', res.resource_id)
    self.assertEqual('AResource', res.name)
    self.assertEqual('COMPLETE', res.status)
    self.assertEqual('ADOPT', res.action)
    self.assertEqual((self.stack.ADOPT, self.stack.COMPLETE), self.stack.state)
    loaded_stack = stack.Stack.load(self.ctx, self.stack.id)
    loaded_stack._update_all_resource_data(False, True)
    self.assertEqual('AResource', loaded_stack.outputs['TestOutput'].get_value())
    self.assertIsNone(loaded_stack['AResource']._stored_properties_data)