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
@functools.wraps(test_method)
def bound_param_test(self):
    if isinstance(testcase_params, abc.Mapping):
        return test_method(self, **testcase_params)
    elif _non_string_or_bytes_iterable(testcase_params):
        return test_method(self, *testcase_params)
    else:
        return test_method(self, testcase_params)