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
def compute_loss_and_updates(self, trainable_variables, non_trainable_variables, x, y, sample_weight, training=False, optimizer_variables=None):
    """This method is stateless and is intended for use with jax.grad."""
    kwargs = {}
    if self._call_has_training_arg:
        kwargs['training'] = training
    y_pred, non_trainable_variables, losses = self.stateless_call(trainable_variables, non_trainable_variables, x, return_losses=True, **kwargs)
    var_mapping = list(zip(self.trainable_variables, trainable_variables))
    var_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables))
    with backend.StatelessScope(state_mapping=var_mapping):
        loss = self.compute_loss(x, y, y_pred, sample_weight, allow_empty=True)
    if losses:
        loss += ops.sum(losses)
    unscaled_loss = loss
    if training and self.optimizer is not None:
        mapping = list(zip(self.optimizer.variables, optimizer_variables))
        with backend.StatelessScope(state_mapping=mapping):
            loss = self.optimizer.scale_loss(loss)
    return (loss, (unscaled_loss, y_pred, non_trainable_variables))