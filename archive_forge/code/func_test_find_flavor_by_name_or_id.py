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
def test_find_flavor_by_name_or_id(self):
    """Tests the find_flavor_by_name_or_id function."""
    flav_id = str(uuid.uuid4())
    flav_name = 'X-Large'
    my_flavor = mock.MagicMock()
    my_flavor.name = flav_name
    my_flavor.id = flav_id
    self.nova_client.flavors.get.side_effect = [my_flavor, nova_exceptions.NotFound(''), nova_exceptions.NotFound('')]
    self.nova_client.flavors.find.side_effect = [my_flavor, nova_exceptions.NotFound('')]
    self.assertEqual(flav_id, self.nova_plugin.find_flavor_by_name_or_id(flav_id))
    self.assertEqual(flav_id, self.nova_plugin.find_flavor_by_name_or_id(flav_name))
    self.assertRaises(nova_exceptions.ClientException, self.nova_plugin.find_flavor_by_name_or_id, 'noflavor')
    self.assertEqual(3, self.nova_client.flavors.get.call_count)
    self.assertEqual(2, self.nova_client.flavors.find.call_count)