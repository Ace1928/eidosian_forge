from collections import OrderedDict
import contextlib
import re
import types
import unittest
from absl.testing import parameterized
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def execute_test_method():
    requested_parameters = tf_inspect.getfullargspec(test_method).args
    for customized_parameter in customized_parameters:
        for argument, value in customized_parameter.modified_arguments(original_kwargs.copy(), requested_parameters).items():
            if value is ParameterModifier.DO_NOT_PASS_TO_THE_TEST:
                kwargs.pop(argument, None)
            else:
                kwargs[argument] = value
    omitted_arguments = set(requested_parameters).difference(set(list(kwargs.keys()) + ['self']))
    if omitted_arguments:
        raise ValueError('The test requires parameters whose arguments were not passed: {} .'.format(omitted_arguments))
    missing_arguments = set(list(kwargs.keys()) + ['self']).difference(set(requested_parameters))
    if missing_arguments:
        raise ValueError('The test does not take parameters that were passed : {} .'.format(missing_arguments))
    kwargs_to_pass = {}
    for parameter in requested_parameters:
        if parameter == 'self':
            kwargs_to_pass[parameter] = self
        else:
            kwargs_to_pass[parameter] = kwargs[parameter]
    test_method(**kwargs_to_pass)