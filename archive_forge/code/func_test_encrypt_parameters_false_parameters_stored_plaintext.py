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
def test_encrypt_parameters_false_parameters_stored_plaintext(self):
    """Test stack loading with disabled parameter value validation."""
    tmpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                description: value1.\n            param2:\n                type: string\n                description: value2.\n                hidden: true\n        resources:\n            a_resource:\n                type: GenericResourceType\n        ')
    env1 = environment.Environment({'param1': 'foo', 'param2': 'bar'})
    self.stack = stack.Stack(self.ctx, 'test', template.Template(tmpl, env=env1))
    cfg.CONF.set_override('encrypt_parameters_and_properties', False)
    self.stack.store()
    db_stack = stack_object.Stack.get_by_id(self.ctx, self.stack.id)
    params = db_stack.raw_template.environment['parameters']
    self.assertEqual('foo', params['param1'])
    self.assertEqual('bar', params['param2'])