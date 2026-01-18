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
def test_load_nested_ok(self):
    self.parent_resource._nested = None
    self.parent_resource.resource_id = 319
    mock_load = self.patchobject(parser.Stack, 'load', return_value='s')
    self.parent_resource.nested()
    mock_load.assert_called_once_with(self.parent_resource.context, self.parent_resource.resource_id)