from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as utils
def supervised_eval_signature_def(inputs, loss, predictions=None, metrics=None):
    return _supervised_signature_def(unexported_constants.SUPERVISED_EVAL_METHOD_NAME, inputs, loss=loss, predictions=predictions, metrics=metrics)