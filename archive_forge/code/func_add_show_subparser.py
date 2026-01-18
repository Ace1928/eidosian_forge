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
def add_show_subparser(subparsers):
    """Add parser for `show`."""
    show_msg = "Usage examples:\nTo show all tag-sets in a SavedModel:\n$saved_model_cli show --dir /tmp/saved_model\n\nTo show all available SignatureDef keys in a MetaGraphDef specified by its tag-set:\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve\n\nFor a MetaGraphDef with multiple tags in the tag-set, all tags must be passed in, separated by ';':\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve,gpu\n\nTo show all inputs and outputs TensorInfo for a specific SignatureDef specified by the SignatureDef key in a MetaGraph.\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve --signature_def serving_default\n\nTo show all ops in a MetaGraph.\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve --list_ops\n\nTo show all available information in the SavedModel:\n$saved_model_cli show --dir /tmp/saved_model --all"
    parser_show = subparsers.add_parser('show', description=show_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_show.set_defaults(func=show)