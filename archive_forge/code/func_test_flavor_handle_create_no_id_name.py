from unittest import mock
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_flavor_handle_create_no_id_name(self):
    self.create_flavor()
    kwargs = {'vcpus': 2, 'disk': 20, 'swap': 2, 'flavorid': 'auto', 'is_public': True, 'rxtx_factor': 1.0, 'ram': 1024, 'ephemeral': 0, 'name': 'm1.xxx'}
    self.patchobject(self.my_flavor, 'physical_resource_name', return_value='m1.xxx')
    value = mock.MagicMock()
    flavor_id = '927202df-1afb-497f-8368-9c2d2f26e5db'
    value.id = flavor_id
    value.is_public = True
    value.get_keys.return_value = {'k': 'v'}
    self.flavors.create.return_value = value
    self.flavors.get.return_value = value
    self.my_flavor.handle_create()
    self.flavors.create.assert_called_once_with(**kwargs)
    value.set_keys.assert_called_once_with({'foo': 'bar'})
    self.assertEqual(flavor_id, self.my_flavor.resource_id)
    self.assertTrue(self.my_flavor.FnGetAtt('is_public'))
    self.assertEqual({'k': 'v'}, self.my_flavor.FnGetAtt('extra_specs'))