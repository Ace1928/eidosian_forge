from unittest import mock
from oslo_config import cfg
import uuid
from heat.db import api as db_api
from heat.db import models
from heat.engine import event
from heat.engine import stack
from heat.engine import template
from heat.objects import event as event_object
from heat.objects import resource_properties_data as rpd_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
def test_load_deprecated_prop_data(self):
    e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'wibble', self.resource._rsrc_prop_data_id, self.resource._stored_properties_data, self.resource.name, self.resource.type())
    e.store()
    with db_api.context_manager.writer.using(self.ctx):
        e_obj = self.ctx.session.get(models.Event, e.id)
        e_obj['resource_properties'] = {'Time': 'not enough'}
        e_obj['rsrc_prop_data'] = None
    ev = event_object.Event.get_all_by_stack(self.ctx, self.stack.id)[0]
    self.assertEqual({'Time': 'not enough'}, ev.resource_properties)