import collections
import os
import time
from tensorflow.python.keras.saving.utils_v1 import export_output as export_output_lib
from tensorflow.python.keras.saving.utils_v1 import mode_keys
from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.keras.saving.utils_v1.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
def export_outputs_for_mode(mode, serving_export_outputs=None, predictions=None, loss=None, metrics=None):
    """Util function for constructing a `ExportOutput` dict given a mode.

  The returned dict can be directly passed to `build_all_signature_defs` helper
  function as the `export_outputs` argument, used for generating a SignatureDef
  map.

  Args:
    mode: A `ModeKeys` specifying the mode.
    serving_export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict or None.
    predictions: A dict of Tensors or single Tensor representing model
        predictions. This argument is only used if serving_export_outputs is not
        set.
    loss: A dict of Tensors or single Tensor representing calculated loss.
    metrics: A dict of (metric_value, update_op) tuples, or a single tuple.
      metric_value must be a Tensor, and update_op must be a Tensor or Op

  Returns:
    Dictionary mapping the a key to an `tf.estimator.export.ExportOutput` object
    The key is the expected SignatureDef key for the mode.

  Raises:
    ValueError: if an appropriate ExportOutput cannot be found for the mode.
  """
    if mode not in SIGNATURE_KEY_MAP:
        raise ValueError('Export output type not found for mode: {}. Expected one of: {}.\nOne likely error is that V1 Estimator Modekeys were somehow passed to this function. Please ensure that you are using the new ModeKeys.'.format(mode, SIGNATURE_KEY_MAP.keys()))
    signature_key = SIGNATURE_KEY_MAP[mode]
    if mode_keys.is_predict(mode):
        return get_export_outputs(serving_export_outputs, predictions)
    elif mode_keys.is_train(mode):
        return {signature_key: export_output_lib.TrainOutput(loss=loss, predictions=predictions, metrics=metrics)}
    else:
        return {signature_key: export_output_lib.EvalOutput(loss=loss, predictions=predictions, metrics=metrics)}