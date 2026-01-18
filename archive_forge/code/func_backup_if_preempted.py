import os
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.distribute import distributed_file_utils
from keras.src.utils import mode_keys
from keras.src.distribute.distributed_file_utils import (
def backup_if_preempted(self):
    if self._enable_save_before_preemption:
        self.preemption_handler._run_counter += 1
        self.preemption_handler._check_preemption_and_maybe_checkpoint()