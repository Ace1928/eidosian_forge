from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_instance_check_not_active(self):
    res = self._get_db_instance()
    self.fake_instance.status = 'FOOBAR'
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.check))
    self.assertIn('FOOBAR', str(exc))
    self.assertEqual((res.CHECK, res.FAILED), res.state)
    self.fake_instance.status = 'ACTIVE'