from unittest import mock
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_flavor_handle_update_keys(self):
    self.create_flavor()
    value = mock.MagicMock()
    self.flavors.get.return_value = value
    value.get_keys.return_value = {}
    new_keys = {'new_foo': 'new_bar'}
    prop_diff = {'extra_specs': new_keys}
    self.my_flavor.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    value.unset_keys.assert_called_once_with({})
    value.set_keys.assert_called_once_with(new_keys)