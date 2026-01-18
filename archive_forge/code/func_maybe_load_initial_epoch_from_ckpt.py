import os
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
def maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
    """Maybe load initial epoch from ckpt considering possible worker recovery.

    When `_ckpt_saved_epoch` attribute exists and is not
    `CKPT_SAVED_EPOCH_UNUSED_VALUE`, this is under multi-worker training setting
    and indicates the worker is recovering from previous failure. In this case,
    infer `initial_epoch` from `self._ckpt_saved_epoch` to continue previous
    unfinished training from certain epoch.

    Args:
      initial_epoch: The original initial_epoch user passes in in `fit()`.
      mode: The mode for running `model.fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """
    epoch = backend.eval(self._ckpt_saved_epoch)
    if mode == mode_keys.ModeKeys.TRAIN and epoch >= 0:
        return epoch + 1
    return initial_epoch