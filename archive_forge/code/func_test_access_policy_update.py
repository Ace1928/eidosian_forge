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
def test_access_policy_update(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'R1': {'Type': 'GenericResourceType'}, 'Policy': {'Type': 'OS::Heat::AccessPolicy', 'Properties': {'AllowedResources': ['R1']}}}}
    self.stack = stack.Stack(self.ctx, 'update_stack_access_policy_test', template.Template(tmpl))
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    tmpl2 = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'R1': {'Type': 'GenericResourceType'}, 'R2': {'Type': 'GenericResourceType'}, 'Policy': {'Type': 'OS::Heat::AccessPolicy', 'Properties': {'AllowedResources': ['R1', 'R2']}}}}
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2))
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)