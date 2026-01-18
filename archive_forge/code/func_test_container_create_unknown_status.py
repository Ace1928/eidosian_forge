import copy
from unittest import mock
from oslo_config import cfg
from zunclient import exceptions as zc_exc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import zun
from heat.engine.resources.openstack.zun import container
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_container_create_unknown_status(self):
    c = self._create_resource('container', self.rsrc_defn, self.stack, status='FOO')
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(c.create))
    self.assertEqual((c.CREATE, c.FAILED), c.state)
    self.assertIn('Unknown status Container', str(exc))