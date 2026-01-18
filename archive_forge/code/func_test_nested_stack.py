import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_nested_stack(self):
    db_api.stack_update(self.ctx, self.stack.id, {'status': 'IN_PROGRESS'})
    child = create_stack(self.ctx, self.template, self.user_creds, owner_id=self.stack.id)
    grandchild = create_stack(self.ctx, self.template, self.user_creds, owner_id=child.id, status='IN_PROGRESS')
    resource = create_resource(self.ctx, grandchild, status='IN_PROGRESS', engine_id=UUID2)
    db_api.reset_stack_status(self.ctx, self.stack.id)
    grandchild = db_api.stack_get(self.ctx, grandchild.id)
    stack = db_api.stack_get(self.ctx, self.stack.id)
    resource = db_api.resource_get(self.ctx, resource.id)
    self.assertEqual('FAILED', grandchild.status)
    self.assertEqual('FAILED', resource.status)
    self.assertIsNone(resource.engine_id)
    self.assertEqual('FAILED', stack.status)