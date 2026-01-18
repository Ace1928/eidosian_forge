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
def test_create_key_with_type(self):
    """Test basic create."""
    key_name = 'with_type'
    tp_test, created_key = self._get_mock_kp_for_create(key_name, key_type='ssh')
    scheduler.TaskRunner(tp_test.create)()
    self.assertEqual((tp_test.CREATE, tp_test.COMPLETE), tp_test.state)
    self.assertEqual(tp_test.resource_id, created_key.name)
    self.fake_keypairs.create.assert_called_once_with(name=key_name, public_key=None, key_type='ssh')
    self.cp_mock.assert_called_once_with()