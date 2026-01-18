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
def test_format_resource_attributes_show_attribute_with_attr(self):
    res = self.stack['generic3']
    res.resource_id = 'generic3_id'
    formatted_attributes = api.format_resource_attributes(res, with_attr=['c'])
    self.assertEqual(4, len(formatted_attributes))
    self.assertIn('foo', formatted_attributes)
    self.assertIn('Foo', formatted_attributes)
    self.assertIn('Another', formatted_attributes)
    self.assertIn('c', formatted_attributes)