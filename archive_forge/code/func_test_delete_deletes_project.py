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
def test_delete_deletes_project(self):
    fkc = fake_ks.FakeKeystoneClient()
    fkc.delete_stack_domain_project = mock.Mock()
    mock_kcp = self.patchobject(keystone.KeystoneClientPlugin, '_create', return_value=fkc)
    self.stack = stack.Stack(self.ctx, 'delete_trust', self.tmpl)
    stack_id = self.stack.store()
    self.stack.set_stack_user_project_id(project_id='aproject456')
    db_s = stack_object.Stack.get_by_id(self.ctx, stack_id)
    self.assertIsNotNone(db_s)
    self.stack.delete()
    mock_kcp.assert_called_with()
    db_s = stack_object.Stack.get_by_id(self.ctx, stack_id)
    self.assertIsNone(db_s)
    self.assertEqual((stack.Stack.DELETE, stack.Stack.COMPLETE), self.stack.state)
    fkc.delete_stack_domain_project.assert_called_once_with(project_id='aproject456')