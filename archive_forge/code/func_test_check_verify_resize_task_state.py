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
def test_check_verify_resize_task_state(self):
    """Tests the check_verify_resize function with resize task_state."""
    my_server = mock.MagicMock(status='Foo')
    setattr(my_server, 'OS-EXT-STS:task_state', 'resize_finish')
    self.nova_client.servers.get.side_effect = [my_server]
    self.assertEqual(False, self.nova_plugin.check_verify_resize('my_server'))