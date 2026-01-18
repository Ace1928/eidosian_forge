from unittest import mock
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_flavor_show_resource(self):
    self.create_flavor()
    self.my_flavor.resource_id = 'flavor_test_id'
    self.my_flavor.client = mock.MagicMock()
    flavors = mock.MagicMock()
    flavor = mock.MagicMock()
    flavor.to_dict.return_value = {'flavor': 'info'}
    flavors.get.return_value = flavor
    self.my_flavor.client().flavors = flavors
    self.assertEqual({'flavor': 'info'}, self.my_flavor.FnGetAtt('show'))
    flavors.get.assert_called_once_with('flavor_test_id')