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
def test_format_stack_resource_with_nested_stack(self):
    res = self.stack['generic4']
    nested_id = {'foo': 'bar'}
    res.has_nested = mock.Mock()
    res.has_nested.return_value = True
    res.nested_identifier = mock.Mock()
    res.nested_identifier.return_value = nested_id
    formatted = api.format_stack_resource(res, False)
    self.assertEqual(nested_id, formatted[rpc_api.RES_NESTED_STACK_ID])