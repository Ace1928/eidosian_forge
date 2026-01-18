import distutils.spawn
import enum
import hashlib
import os as _os
import platform as _platform
import subprocess as _subprocess
import tempfile as _tempfile
from typing import Optional
import warnings
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_options_pb2 as quant_opts_pb2
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import util
from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import ConverterError
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper as _metrics_wrapper
from tensorflow.lite.toco import model_flags_pb2 as _model_flags_pb2
from tensorflow.lite.toco import toco_flags_pb2 as _conversion_flags_pb2
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export as _tf_export
@convert_phase(Component.CONVERT_TF_TO_TFLITE_MODEL, SubComponent.CONVERT_GRAPHDEF)
def convert_graphdef_with_arrays(input_data, input_arrays_with_shape, output_arrays, control_output_arrays, **kwargs):
    """Convert a frozen GraphDef that can't be loaded in TF.

  Conversion can be customized by providing arguments that are forwarded to
  `build_model_flags` and `build_conversion_flags` (see documentation).

  Args:
    input_data: Input data (i.e. often `sess.graph_def`),
    input_arrays_with_shape: Tuple of strings representing input tensor names
      and list of integers representing input shapes (e.g., [("foo" : [1, 16,
      16, 3])]). Use only when graph cannot be loaded into TensorFlow and when
      `input_tensors` is None.
    output_arrays: List of output tensors to freeze graph with. Use only when
      graph cannot be loaded into TensorFlow and when `output_tensors` is None.
    control_output_arrays: Control output node names. This is used when
      converting a Graph with no output tensors. For example, if the graph's
      last operation is a Print op, just specify that op's name in this field.
      This can be used together with the `output_arrays` parameter.
    **kwargs: See `build_model_flags` and `build_conversion_flags`.

  Returns:
    The converted data. For example if TFLite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    Defined in `build_conversion_flags`.
  """
    model_flags = build_model_flags(**kwargs)
    conversion_flags = build_conversion_flags(**kwargs)
    enable_mlir_converter = kwargs.get('enable_mlir_converter', True)
    quantized_input_stats = kwargs.get('quantized_input_stats', None)
    for idx, (name, shape) in enumerate(input_arrays_with_shape):
        input_array = model_flags.input_arrays.add()
        if _is_quantized_input_stats_required(conversion_flags):
            if quantized_input_stats:
                input_array.mean_value, input_array.std_value = quantized_input_stats[idx]
            else:
                raise ValueError('The `quantized_input_stats` flag must be defined when either `inference_type` flag or `inference_input_type` flag is set to tf.int8 or tf.uint8.')
        input_array.name = name
        input_array.shape.dims.extend(list(map(int, shape)))
    if output_arrays:
        for name in output_arrays:
            model_flags.output_arrays.append(name)
    if control_output_arrays:
        for name in control_output_arrays:
            model_flags.control_output_arrays.append(name)
    data = convert(model_flags, conversion_flags, input_data.SerializeToString(), debug_info_str=None, enable_mlir_converter=enable_mlir_converter)
    return data