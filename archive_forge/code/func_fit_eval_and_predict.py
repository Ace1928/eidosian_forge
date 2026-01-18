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
def fit_eval_and_predict(initial_weights, input_fn, model_fn, distribution=None, is_stateful_model=False):
    """Generates results for fit/predict/evaluate for given model."""
    training_inputs, eval_inputs, predict_inputs = input_fn()
    model = model_fn(initial_weights=initial_weights, distribution=distribution, input_shapes=get_shapes(training_inputs['x']))
    result = {}
    result['training_history_1'] = model.fit(**training_inputs).history
    if eval_inputs is not None:
        result['eval_result_1'] = model.evaluate(**eval_inputs)
    result['weights_1'] = model.get_weights()
    if predict_inputs is not None:
        predict_length = 1
        if is_stateful_model:
            predict_length = 3
        for i in range(predict_length):
            result_key = f'predict_result_{i}'
            result[result_key] = model.predict(**predict_inputs)
    result['training_history_2'] = model.fit(**training_inputs).history
    if eval_inputs is not None:
        result['eval_result_2'] = model.evaluate(**eval_inputs)
    result['weights_2'] = model.get_weights()
    return result