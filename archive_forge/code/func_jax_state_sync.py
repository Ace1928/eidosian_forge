import collections
import itertools
from functools import partial
import jax
import numpy as np
import tree
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import ops
from keras.src import optimizers as optimizers_module
from keras.src.backend import distribution_lib as jax_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def jax_state_sync(self):
    if not getattr(self, '_jax_state', None) or self._jax_state_synced:
        return
    trainable_variables = self._jax_state.get('trainable_variables', None)
    non_trainable_variables = self._jax_state.get('non_trainable_variables', None)
    optimizer_variables = self._jax_state.get('optimizer_variables', None)
    metrics_variables = self._jax_state.get('metrics_variables', None)
    if trainable_variables:
        for ref_v, v in zip(self.trainable_variables, trainable_variables):
            ref_v.assign(v)
    if non_trainable_variables:
        for ref_v, v in zip(self.non_trainable_variables, non_trainable_variables):
            ref_v.assign(v)
    if optimizer_variables:
        for ref_v, v in zip(self.optimizer.variables, optimizer_variables):
            ref_v.assign(v)
    if metrics_variables:
        for ref_v, v in zip(self.metrics_variables, metrics_variables):
            ref_v.assign(v)
    self._jax_state_synced = True