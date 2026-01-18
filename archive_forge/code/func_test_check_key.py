import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_check_key(self):
    res = self._get_test_resource(self.kp_template)
    res.state_set(res.CREATE, res.COMPLETE, 'for test')
    res.client = mock.Mock()
    scheduler.TaskRunner(res.check)()
    self.assertEqual((res.CHECK, res.COMPLETE), res.state)