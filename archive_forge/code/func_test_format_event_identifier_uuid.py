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
def test_format_event_identifier_uuid(self):
    event = self._dummy_event()
    event_keys = set((rpc_api.EVENT_ID, rpc_api.EVENT_STACK_ID, rpc_api.EVENT_STACK_NAME, rpc_api.EVENT_TIMESTAMP, rpc_api.EVENT_RES_NAME, rpc_api.EVENT_RES_PHYSICAL_ID, rpc_api.EVENT_RES_ACTION, rpc_api.EVENT_RES_STATUS, rpc_api.EVENT_RES_STATUS_DATA, rpc_api.EVENT_RES_TYPE, rpc_api.EVENT_RES_PROPERTIES))
    formatted = api.format_event(event, self.stack.identifier())
    self.assertEqual(event_keys, set(formatted.keys()))
    event_id_formatted = formatted[rpc_api.EVENT_ID]
    self.assertEqual({'path': '/resources/generic1/events/%s' % event.uuid, 'stack_id': self.stack.id, 'stack_name': 'test_stack', 'tenant': 'test_tenant_id'}, event_id_formatted)