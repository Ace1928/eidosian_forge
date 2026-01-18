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
def back_up(self, epoch):
    """Back up the current state of training into a checkpoint file.

    Args:
      epoch: The current epoch information to be saved.
    """
    backend.set_value(self._ckpt_saved_epoch, epoch)
    if self.write_checkpoint_manager.save():
        distributed_file_utils.remove_temp_dirpath(self.write_checkpoint_manager.directory, self._model.distribute_strategy)