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
def test_format_stack_updated_time(self):
    self.stack.updated_time = None
    info = api.format_stack(self.stack)
    self.assertIsNone(info['updated_time'])
    self.stack.updated_time = datetime(1970, 1, 1)
    info = api.format_stack(self.stack)
    self.assertEqual('1970-01-01T00:00:00Z', info['updated_time'])