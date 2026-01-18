import argparse
import platform
import ast
import os
import re
from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
import numpy as np
from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc
def preprocess_input_examples_arg_string(input_examples_str):
    """Parses input into dict that maps input keys to lists of tf.Example.

  Parses input string in the format of 'input_key1=[{feature_name:
  feature_list}];input_key2=[{feature_name:feature_list}];' into a dictionary
  that maps each input_key to its list of serialized tf.Example.

  Args:
    input_examples_str: A string that specifies a list of dictionaries of
    feature_names and their feature_lists for each input.
    Each input is separated by semicolon. For each input key:
      'input=[{feature_name1: feature_list1, feature_name2:feature_list2}]'
      items in feature_list can be the type of float, int, long or str.

  Returns:
    A dictionary that maps input keys to lists of serialized tf.Example.

  Raises:
    ValueError: An error when the given tf.Example is not a list.
  """
    input_dict = preprocess_input_exprs_arg_string(input_examples_str)
    for input_key, example_list in input_dict.items():
        if not isinstance(example_list, list):
            raise ValueError('tf.Example input must be a list of dictionaries, but "%s" is %s' % (example_list, type(example_list)))
        input_dict[input_key] = [_create_example_string(example) for example in example_list]
    return input_dict