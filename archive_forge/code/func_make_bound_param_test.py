from the command line::
from collections import abc
import functools
import inspect
import itertools
import re
import types
import unittest
import warnings
from absl.testing import absltest
def make_bound_param_test(testcase_params):

    @functools.wraps(test_method)
    def bound_param_test(self):
        if isinstance(testcase_params, abc.Mapping):
            return test_method(self, **testcase_params)
        elif _non_string_or_bytes_iterable(testcase_params):
            return test_method(self, *testcase_params)
        else:
            return test_method(self, testcase_params)
    if naming_type is _NAMED:
        bound_param_test.__x_use_name__ = True
        testcase_name = None
        if isinstance(testcase_params, abc.Mapping):
            if _NAMED_DICT_KEY not in testcase_params:
                raise RuntimeError('Dict for named tests must contain key "%s"' % _NAMED_DICT_KEY)
            testcase_name = testcase_params[_NAMED_DICT_KEY]
            testcase_params = {k: v for k, v in testcase_params.items() if k != _NAMED_DICT_KEY}
        elif _non_string_or_bytes_iterable(testcase_params):
            if not isinstance(testcase_params[0], str):
                raise RuntimeError('The first element of named test parameters is the test name suffix and must be a string')
            testcase_name = testcase_params[0]
            testcase_params = testcase_params[1:]
        else:
            raise RuntimeError('Named tests must be passed a dict or non-string iterable.')
        test_method_name = self._original_name
        if test_method_name.startswith('test_') and testcase_name and (not testcase_name.startswith('_')):
            test_method_name += '_'
        bound_param_test.__name__ = test_method_name + str(testcase_name)
    elif naming_type is _ARGUMENT_REPR:
        if isinstance(testcase_params, types.GeneratorType):
            testcase_params = tuple(testcase_params)
        params_repr = '(%s)' % (_format_parameter_list(testcase_params),)
        bound_param_test.__x_params_repr__ = params_repr
    else:
        raise RuntimeError('%s is not a valid naming type.' % (naming_type,))
    bound_param_test.__doc__ = '%s(%s)' % (bound_param_test.__name__, _format_parameter_list(testcase_params))
    if test_method.__doc__:
        bound_param_test.__doc__ += '\n%s' % (test_method.__doc__,)
    if inspect.iscoroutinefunction(test_method):
        return _async_wrapped(bound_param_test)
    return bound_param_test