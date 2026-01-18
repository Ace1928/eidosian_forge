from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_define_method_is_actually_called(self):
    checker = mock.MagicMock()

    @api_versions.wraps('2.2', '2.6')
    def some_func(*args, **kwargs):
        checker(*args, **kwargs)
    obj = self._get_obj_with_vers('2.4')
    some_args = ('arg_1', 'arg_2')
    some_kwargs = {'key1': 'value1', 'key2': 'value2'}
    some_func(obj, *some_args, **some_kwargs)
    checker.assert_called_once_with(*(obj,) + some_args, **some_kwargs)