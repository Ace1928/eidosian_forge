from keystone.auth import plugins
from keystone.tests import unit
def test_convert_methods_to_integer(self):
    auth_methods = ['password', 'token', 'totp']
    self.config_fixture.config(group='auth', methods=auth_methods)
    method_integer = plugins.convert_method_list_to_integer(['password'])
    self.assertEqual(1, method_integer)
    method_integer = plugins.convert_method_list_to_integer(['password', 'token'])
    self.assertEqual(3, method_integer)
    method_integer = plugins.convert_method_list_to_integer(['password', 'totp'])
    self.assertEqual(5, method_integer)
    method_integer = plugins.convert_method_list_to_integer(['token', 'totp'])
    self.assertEqual(6, method_integer)
    method_integer = plugins.convert_method_list_to_integer(['password', 'token', 'totp'])
    self.assertEqual(7, method_integer)