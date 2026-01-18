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
def test_delete_trust_trustor(self):
    self.stub_keystoneclient(user_id='thetrustor')
    trustor_ctx = utils.dummy_context(user_id='thetrustor')
    self.stack = stack.Stack(trustor_ctx, 'delete_trust_nt', self.tmpl)
    stack_id = self.stack.store()
    db_s = stack_object.Stack.get_by_id(self.ctx, stack_id)
    self.assertIsNotNone(db_s)
    user_creds_id = db_s.user_creds_id
    self.assertIsNotNone(user_creds_id)
    user_creds = ucreds_object.UserCreds.get_by_id(self.ctx, user_creds_id)
    self.assertEqual('thetrustor', user_creds.get('trustor_user_id'))
    self.stack.delete()
    db_s = stack_object.Stack.get_by_id(trustor_ctx, stack_id)
    self.assertIsNone(db_s)
    self.assertEqual((stack.Stack.DELETE, stack.Stack.COMPLETE), self.stack.state)