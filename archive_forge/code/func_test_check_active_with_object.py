import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_check_active_with_object(self):
    if self.e_raise:
        self.assertRaises(self.e_raise, self.nova_plugin._check_active, self.server)
        self.r_mock.assert_called_once_with(self.server)
    elif self.status in self.nova_plugin.deferred_server_statuses:
        self.assertFalse(self.nova_plugin._check_active(self.server))
        self.r_mock.assert_called_once_with(self.server)
    else:
        self.assertTrue(self.nova_plugin._check_active(self.server))
        self.assertEqual(0, self.r_mock.call_count)
    self.assertEqual(0, self.f_mock.call_count)