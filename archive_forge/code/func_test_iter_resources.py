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
@mock.patch.object(resource_objects.Resource, 'get_all_by_stack')
def test_iter_resources(self, mock_db_call):
    tpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'A': {'Type': 'GenericResourceType'}, 'B': {'Type': 'GenericResourceType'}}}
    self.stack = stack.Stack(self.ctx, 'test_stack', template.Template(tpl), status_reason='blarg')
    self.stack.store()
    mock_rsc_a = mock.MagicMock(current_template_id=self.stack.t.id)
    mock_rsc_a.name = 'A'
    mock_rsc_b = mock.MagicMock(current_template_id=self.stack.t.id)
    mock_rsc_b.name = 'B'
    mock_db_call.return_value = {'A': mock_rsc_a, 'B': mock_rsc_b}
    all_resources = list(self.stack.iter_resources())
    mock_db_call.assert_called_once_with(self.ctx, self.stack.id)
    names = sorted([r.name for r in all_resources])
    self.assertEqual(['A', 'B'], names)