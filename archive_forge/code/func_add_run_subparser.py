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
def add_run_subparser(subparsers):
    """Add parser for `run`."""
    run_msg = 'Usage example:\nTo run input tensors from files through a MetaGraphDef and save the output tensors to files:\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve \\\n   --signature_def serving_default \\\n   --inputs input1_key=/tmp/124.npz[x],input2_key=/tmp/123.npy \\\n   --input_exprs \'input3_key=np.ones(2)\' \\\n   --input_examples \'input4_key=[{"id":[26],"weights":[0.5, 0.5]}]\' \\\n   --outdir=/out\n\nFor more information about input file format, please see:\nhttps://www.tensorflow.org/guide/saved_model_cli\n'
    parser_run = subparsers.add_parser('run', description=run_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_run.set_defaults(func=run)