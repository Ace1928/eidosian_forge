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
def convert_with_tensorrt():
    """Function triggered by 'convert tensorrt' command."""
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    if not _SMCLI_CONVERT_TF1_MODEL.value:
        params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(max_workspace_size_bytes=_SMCLI_MAX_WORKSPACE_SIZE_BYTES.value, precision_mode=_SMCLI_PRECISION_MODE.value, minimum_segment_size=_SMCLI_MINIMUM_SEGMENT_SIZE.value)
        try:
            converter = trt.TrtGraphConverterV2(input_saved_model_dir=_SMCLI_DIR.value, input_saved_model_tags=_SMCLI_TAG_SET.value.split(','), **params._asdict())
            converter.convert()
        except Exception as exc:
            raise RuntimeError('{}. Try passing "--convert_tf1_model=True".'.format(exc)) from exc
        converter.save(output_saved_model_dir=_SMCLI_OUTPUT_DIR.value)
    else:
        trt.create_inference_graph(None, None, max_batch_size=1, max_workspace_size_bytes=_SMCLI_MAX_WORKSPACE_SIZE_BYTES.value, precision_mode=_SMCLI_PRECISION_MODE.value, minimum_segment_size=_SMCLI_MINIMUM_SEGMENT_SIZE.value, is_dynamic_op=True, input_saved_model_dir=_SMCLI_DIR.value, input_saved_model_tags=_SMCLI_TAG_SET.value.split(','), output_saved_model_dir=_SMCLI_OUTPUT_DIR.value)