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
@mock.patch.object(stack.Stack, 'db_resource_get')
def test_lightweight_stack_getrefid(self, mock_drg):
    tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'foo'}}}}})
    rsrcs_data = {'foo': {'reference_id': 'physical-resource-id', 'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE'}, 'bar': {'reference_id': 'bar-id', 'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE'}}
    cache_data = {n: node_data.NodeData.from_dict(d) for n, d in rsrcs_data.items()}
    tmpl_stack = stack.Stack(self.ctx, 'test', tmpl)
    tmpl_stack.store()
    lightweight_stack = stack.Stack.load(self.ctx, stack_id=tmpl_stack.id, cache_data=cache_data)
    bar = resource.Resource('bar', lightweight_stack.defn.resource_definition('bar'), lightweight_stack)
    self.assertEqual('physical-resource-id', bar.properties['Foo'])
    resource_id = lightweight_stack.defn['foo'].FnGetRefId()
    self.assertEqual('physical-resource-id', resource_id)
    self.assertFalse(mock_drg.called)