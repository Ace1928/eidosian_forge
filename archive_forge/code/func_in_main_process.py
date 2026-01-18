import collections
import copy
import re
import sys
import types
import unittest
from absl import app
import six
from tensorflow.python.client import session
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.distribute.combinations.in_main_process', v1=[])
def in_main_process():
    """Whether it's in the main test process.

  This is normally used to prepare the test environment which should only happen
  in the main process.

  Returns:
    A boolean.
  """
    return not _running_in_worker