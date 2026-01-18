from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
@mock.patch('zunclient.api_versions._get_function_name')
@mock.patch('zunclient.api_versions.VersionedMethod')
def test_api_version_doesnt_match(self, mock_versioned_method, mock_name):
    func_name = 'foo'
    mock_name.return_value = func_name
    mock_versioned_method.side_effect = self._side_effect_of_vers_method

    @api_versions.wraps('2.2', '2.6')
    def foo(*args, **kwargs):
        pass
    self.assertRaises(exceptions.VersionNotFoundForAPIMethod, foo, self._get_obj_with_vers('2.1'))
    mock_versioned_method.assert_called_once_with(func_name, api_versions.APIVersion('2.2'), api_versions.APIVersion('2.6'), mock.ANY)