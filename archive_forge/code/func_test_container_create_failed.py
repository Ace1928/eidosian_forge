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
def test_container_create_failed(self):
    cfg.CONF.set_override('action_retry_limit', 0)
    c = self._create_resource('container', self.rsrc_defn, self.stack, status='Error')
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(c.create))
    self.assertEqual((c.CREATE, c.FAILED), c.state)
    self.assertIn('Error in creating container ', str(exc))