import contextlib
import warnings
import numpy as np
import tensorflow as tf
import tree
from packaging.version import Version
from tensorflow.python.eager import context as tf_context
from keras.src import callbacks as callbacks_module
from keras.src import metrics as metrics_module
from keras.src import optimizers as optimizers_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def enumerate_epoch(self):
    if self.steps_per_epoch:
        if not self._current_iterator:
            self._current_iterator = iter(self._distributed_dataset)
        for step in range(0, self.steps_per_epoch, self.steps_per_execution):
            yield (step, self._current_iterator)
    else:
        iterator = iter(self._distributed_dataset)
        if self.num_batches:
            for step in range(0, self.num_batches, self.steps_per_execution):
                yield (step, iterator)
        else:
            step = -1
            while True:
                step += self.steps_per_execution
                self._steps_seen = step + 1
                yield (step, iterator)
    self.data_adapter.on_epoch_end()