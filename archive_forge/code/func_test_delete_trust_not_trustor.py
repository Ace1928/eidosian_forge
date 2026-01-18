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
def test_delete_trust_not_trustor(self):
    trustor_ctx = utils.dummy_context(user_id='thetrustor')
    other_ctx = utils.dummy_context(user_id='nottrustor')
    stored_ctx = utils.dummy_context(trust_id='thetrust')
    mock_kc = self.patchobject(hkc, 'KeystoneClient')
    self.stub_keystoneclient(user_id='thetrustor')
    mock_sc = self.patchobject(stack.Stack, 'stored_context')
    mock_sc.return_value = stored_ctx
    self.stack = stack.Stack(trustor_ctx, 'delete_trust_nt', self.tmpl)
    stack_id = self.stack.store()
    db_s = stack_object.Stack.get_by_id(self.ctx, stack_id)
    self.assertIsNotNone(db_s)
    user_creds_id = db_s.user_creds_id
    self.assertIsNotNone(user_creds_id)
    user_creds = ucreds_object.UserCreds.get_by_id(self.ctx, user_creds_id)
    self.assertEqual('thetrustor', user_creds.get('trustor_user_id'))
    mock_kc.return_value = fake_ks.FakeKeystoneClient(user_id='nottrustor')
    loaded_stack = stack.Stack.load(other_ctx, self.stack.id)
    loaded_stack.delete()
    mock_sc.assert_called_with()
    db_s = stack_object.Stack.get_by_id(other_ctx, stack_id)
    self.assertIsNone(db_s)
    self.assertEqual((stack.Stack.DELETE, stack.Stack.COMPLETE), loaded_stack.state)