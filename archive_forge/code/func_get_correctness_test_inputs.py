import functools
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src.distribute import distributed_training_utils
from keras.src.distribute.strategy_combinations import all_strategies
from keras.src.distribute.strategy_combinations import (
from keras.src.distribute.strategy_combinations import strategies_minus_tpu
from keras.src.mixed_precision import policy
from keras.src.utils import data_utils
def get_correctness_test_inputs(use_numpy, use_validation_data, with_distribution, x_train, y_train, x_eval, y_eval, x_predict, training_epochs):
    """Generates the inputs for correctness check when enable Keras with DS."""
    global_batch_size = _GLOBAL_BATCH_SIZE
    batch_size = get_batch_size(global_batch_size, with_distribution)
    if use_numpy:
        training_inputs = {'batch_size': batch_size, 'x': x_train, 'y': y_train, 'epochs': training_epochs, 'shuffle': False}
        if use_validation_data:
            eval_inputs = None
            training_inputs['validation_data'] = (x_eval, y_eval)
        else:
            eval_inputs = {'batch_size': batch_size, 'x': x_eval, 'y': y_eval}
        predict_inputs = {'x': x_predict}
    else:
        training_data_size = get_data_size(x_train)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        x = batch_wrapper(train_dataset, batch_size, repeat=training_epochs)
        steps_per_epoch = int(np.ceil(1.0 * training_data_size / global_batch_size))
        training_inputs = {'batch_size': None, 'x': x, 'y': None, 'epochs': training_epochs, 'shuffle': False, 'steps_per_epoch': steps_per_epoch}
        if use_validation_data:
            eval_inputs = None
            eval_dataset = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
            x = batch_wrapper(eval_dataset, batch_size)
            training_inputs['validation_data'] = x
            training_inputs['validation_steps'] = 5
        else:
            eval_dataset = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
            x = batch_wrapper(eval_dataset, batch_size)
            eval_steps = int(np.ceil(1.0 * get_data_size(x_eval) / global_batch_size))
            eval_inputs = {'batch_size': None, 'x': x, 'y': None, 'steps': eval_steps}
        predict_batch_size = get_batch_size(get_data_size(x_predict), with_distribution)
        predict_dataset = tf.data.Dataset.from_tensor_slices(x_predict)
        predict_dataset = batch_wrapper(predict_dataset, predict_batch_size)
        predict_inputs = {'steps': 1, 'x': predict_dataset}
    return (training_inputs, eval_inputs, predict_inputs)