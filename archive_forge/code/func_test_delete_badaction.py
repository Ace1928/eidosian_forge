import copy
import time
from unittest import mock
import fixtures
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_delete_badaction(self):
    self.stack = stack.Stack(self.ctx, 'delete_badaction_test', self.tmpl)
    stack_id = self.stack.store()
    db_s = stack_object.Stack.get_by_id(self.ctx, stack_id)
    self.assertIsNotNone(db_s)
    self.stack.delete(action='wibble')
    db_s = stack_object.Stack.get_by_id(self.ctx, stack_id)
    self.assertIsNotNone(db_s)
    self.assertEqual((stack.Stack.DELETE, stack.Stack.FAILED), self.stack.state)