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
def test_container_get_live_state(self):
    c = self._create_resource('container', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(c.create)()
    self._mock_get_client()
    reality = c.get_live_state(c.properties)
    self.assertEqual({container.Container.NAME: self.fake_name, container.Container.CPU: self.fake_cpu, container.Container.MEMORY: self.fake_memory}, reality)