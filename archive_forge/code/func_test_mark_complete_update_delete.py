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
@mock.patch.object(stack.Stack, 'purge_db')
def test_mark_complete_update_delete(self, mock_purge_db):
    tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Description': 'Empty Template'})
    cfg.CONF.set_default('convergence_engine', True)
    tmpl_stack = stack.Stack(self.ctx, 'test', tmpl, convergence=True)
    tmpl_stack.prev_raw_template_id = 1
    tmpl_stack.action = tmpl_stack.DELETE
    tmpl_stack.status = tmpl_stack.IN_PROGRESS
    tmpl_stack.current_traversal = 'some-traversal'
    tmpl_stack.store()
    tmpl_stack.mark_complete()
    self.assertTrue(mock_purge_db.called)