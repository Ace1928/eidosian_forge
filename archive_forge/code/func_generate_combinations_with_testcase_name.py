import collections
import contextlib
import functools
import itertools
import threading
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator
def generate_combinations_with_testcase_name(**kwargs):
    """Generate combinations based on its keyword arguments using combine().

  This function calls combine() and appends a testcase name to the list of
  dictionaries returned. The 'testcase_name' key is a required for named
  parameterized tests.

  Args:
    **kwargs: keyword arguments of form `option=[possibilities, ...]` or
      `option=the_only_possibility`.

  Returns:
    a list of dictionaries for each combination. Keys in the dictionaries are
    the keyword argument names.  Each key has one value - one of the
    corresponding keyword argument values.
  """
    sort_by_key = lambda k: k[0]
    combinations = []
    for key, values in sorted(kwargs.items(), key=sort_by_key):
        if not isinstance(values, list):
            values = [values]
        combinations.append([(key, value) for value in values])
    combinations = [collections.OrderedDict(result) for result in itertools.product(*combinations)]
    named_combinations = []
    for combination in combinations:
        assert isinstance(combination, collections.OrderedDict)
        name = ''.join(['_{}_{}'.format(''.join(filter(str.isalnum, key)), ''.join(filter(str.isalnum, str(value)))) for key, value in combination.items()])
        named_combinations.append(collections.OrderedDict(list(combination.items()) + [('testcase_name', '_test{}'.format(name))]))
    return named_combinations