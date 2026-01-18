import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
def test_format_stack_resource(self):
    self.stack.created_time = datetime(2015, 8, 3, 17, 5, 1)
    self.stack.updated_time = datetime(2015, 8, 3, 17, 6, 2)
    res = self.stack['generic1']
    resource_keys = set((rpc_api.RES_CREATION_TIME, rpc_api.RES_UPDATED_TIME, rpc_api.RES_NAME, rpc_api.RES_PHYSICAL_ID, rpc_api.RES_ACTION, rpc_api.RES_STATUS, rpc_api.RES_STATUS_DATA, rpc_api.RES_TYPE, rpc_api.RES_ID, rpc_api.RES_STACK_ID, rpc_api.RES_STACK_NAME, rpc_api.RES_REQUIRED_BY))
    resource_details_keys = resource_keys.union(set((rpc_api.RES_DESCRIPTION, rpc_api.RES_METADATA, rpc_api.RES_ATTRIBUTES)))
    formatted = api.format_stack_resource(res, True)
    self.assertEqual(resource_details_keys, set(formatted.keys()))
    formatted = api.format_stack_resource(res, False)
    self.assertEqual(resource_keys, set(formatted.keys()))
    self.assertEqual(heat_timeutils.isotime(self.stack.created_time), formatted[rpc_api.RES_CREATION_TIME])
    self.assertEqual(heat_timeutils.isotime(self.stack.updated_time), formatted[rpc_api.RES_UPDATED_TIME])
    self.assertEqual(res.INIT, formatted[rpc_api.RES_ACTION])