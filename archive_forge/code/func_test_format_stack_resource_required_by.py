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
def test_format_stack_resource_required_by(self):
    res1 = api.format_stack_resource(self.stack['generic1'])
    res2 = api.format_stack_resource(self.stack['generic2'])
    self.assertEqual(['generic2'], res1['required_by'])
    self.assertEqual([], res2['required_by'])