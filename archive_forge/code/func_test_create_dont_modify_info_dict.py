import testtools
from saharaclient.tests.unit import base as test_base
def test_create_dont_modify_info_dict(self):
    dict = {'name': 'test', 'description': 'Changed Description'}
    dict_copy = dict.copy()
    resource = test_base.TestResource(None, dict)
    self.assertIsNotNone(resource)
    self.assertEqual(dict_copy, dict)