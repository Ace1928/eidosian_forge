from collections import OrderedDict
import contextlib
import re
import types
import unittest
from absl.testing import parameterized
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def modified_arguments(self, kwargs, requested_parameters):
    if self._parameter_name in requested_parameters:
        return {}
    else:
        return {self._parameter_name: ParameterModifier.DO_NOT_PASS_TO_THE_TEST}