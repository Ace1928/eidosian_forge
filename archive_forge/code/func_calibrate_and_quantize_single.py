import numpy as np
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.framework import dtypes
from tensorflow.python.util.lazy_loader import LazyLoader
@convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.QUANTIZE_USING_DEPRECATED_QUANTIZER)
def calibrate_and_quantize_single(self, dataset_gen, input_type, output_type, allow_float, op_output_name, resize_input=True):
    """Calibrates the model with specified generator and then quantizes it.

    Only the single op with output op_output_name will be quantized.
    The input shapes of the calibrator are resized with the calibration data.

    Returns:
      A quantized model.

    Args:
      dataset_gen: A generator that generates calibration samples.
      input_type: A tf.dtype representing the desired real-value input type.
      output_type: A tf.dtype representing the desired real-value output type.
      allow_float: A boolean. False if the resulting model cannot perform float
        computation, useful when targeting an integer-only backend. If False, an
        error will be thrown if an operation cannot be quantized, otherwise the
        model will fallback to float ops.
      op_output_name: A string, only this op will be quantized.
      resize_input: A boolean. True if the shape of the sample data is different
        from the input.
    """
    self._feed_tensors(dataset_gen, resize_input)
    return self._calibrator.QuantizeModel(np.dtype(input_type.as_numpy_dtype()).num, np.dtype(output_type.as_numpy_dtype()).num, allow_float, op_output_name)