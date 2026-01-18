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
def test_get_attribute_autoscaling_convg(self):
    t = template_format.parse(heat_autoscaling_group_template)
    tmpl = templatem.Template(t)
    cache_data = {'my_autoscaling_group': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'attrs': {'current_size': 4}})}
    stack = parser.Stack(utils.dummy_context(), 'test_att', tmpl, cache_data=cache_data)
    rsrc = stack.defn['my_autoscaling_group']
    self.assertEqual(4, rsrc.FnGetAtt('current_size'))