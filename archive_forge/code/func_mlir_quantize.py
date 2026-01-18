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
@convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.QUANTIZE)
def mlir_quantize(input_data_str, disable_per_channel=False, fully_quantize=False, inference_type=_types_pb2.QUANTIZED_INT8, input_data_type=dtypes.float32, output_data_type=dtypes.float32, enable_numeric_verify=False, enable_whole_model_verify=False, denylisted_ops=None, denylisted_nodes=None, enable_variable_quantization=False):
    """Quantize `input_data_str` with calibration results.

  Args:
    input_data_str: Input data in serialized form (e.g. a TFLITE model with
      calibration results).
    disable_per_channel: Bool indicating whether to do per-channel or per-tensor
      quantization
    fully_quantize: Bool indicating whether to fully quantize the model. Besides
      model body, the input/output will be quantized as well.
    inference_type: Data type for the activations. The default value is int8.
    input_data_type: Data type for the inputs. The default value is float32.
    output_data_type: Data type for the outputs. The default value is float32.
    enable_numeric_verify: Experimental. Subject to change. Bool indicating
      whether to add NumericVerify ops into the debug mode quantized model.
    enable_whole_model_verify: Experimental. Subject to change. Bool indicating
      whether to add verification for layer by layer, or on whole model. When
      disabled (per-layer) float and quantized ops will be run from same input
      (output of previous quantized layer). When enabled, float and quantized
      ops will run with respective float and quantized output of previous ops.
    denylisted_ops: Experimental. Subject to change. Set of ops to denylist.
    denylisted_nodes: Experimental. Subject to change. Set of notes to denylist.
    enable_variable_quantization: Experimental. Subject to change. Bool
      indicating whether to enable quantization of the residual variables
      remaining after the variable freezing pass.

  Returns:
    Quantized model in serialized form (e.g. a TFLITE model) with floating-point
    inputs and outputs.
  """
    return wrap_toco.wrapped_experimental_mlir_quantize(input_data_str, disable_per_channel, fully_quantize, inference_type, convert_tensor_tf_type_to_tflite_type(input_data_type), convert_tensor_tf_type_to_tflite_type(output_data_type), enable_numeric_verify, enable_whole_model_verify, denylisted_ops, denylisted_nodes, enable_variable_quantization)