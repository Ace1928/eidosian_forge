import collections
import functools
import itertools
import unittest
from absl.testing import parameterized
from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest
def run_with_all_weight_formats(test_or_class=None, exclude_formats=None):
    """Runs all tests with the supported formats for saving weights."""
    exclude_formats = exclude_formats or []
    exclude_formats.append('tf_no_traces')
    return run_with_all_saved_model_formats(test_or_class, exclude_formats)