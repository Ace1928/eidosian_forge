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
def test_get_keypair(self):
    """Tests the get_keypair function."""
    my_pub_key = 'a cool public key string'
    my_key_name = 'mykey'
    my_key = mock.MagicMock()
    my_key.public_key = my_pub_key
    my_key.name = my_key_name
    self.nova_client.keypairs.get.side_effect = [my_key, nova_exceptions.NotFound(404)]
    self.assertEqual(my_key, self.nova_plugin.get_keypair(my_key_name))
    self.assertRaises(exception.EntityNotFound, self.nova_plugin.get_keypair, 'notakey')
    calls = [mock.call(my_key_name), mock.call('notakey')]
    self.nova_client.keypairs.get.assert_has_calls(calls)