from keystone.auth import plugins
from keystone.tests import unit
def test_construct_method_map_with_two_methods(self):
    auth_methods = ['password', 'token']
    self.config_fixture.config(group='auth', methods=auth_methods)
    expected_method_map = {1: 'password', 2: 'token'}
    method_map = plugins.construct_method_map_from_config()
    self.assertDictEqual(expected_method_map, method_map)