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
def test_delete_stack_with_resource_log_is_clear(self):
    debug_logger = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG, format='%(levelname)8s [%(name)s] %(message)s'))
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'GenericResourceType'}}}
    self.stack = stack.Stack(self.ctx, 'delete_log_test', template.Template(tmpl))
    self.stack.store()
    self.stack.create()
    self.assertEqual((self.stack.CREATE, self.stack.COMPLETE), self.stack.state)
    self.stack.delete()
    self.assertNotIn('destroy from None running', debug_logger.output)