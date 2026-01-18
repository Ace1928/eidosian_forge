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
def run_correctness_test(self, distribution, use_numpy, use_validation_data, with_batch_norm=None, is_stateful_model=False, partial_last_batch=None, training_epochs=2):
    with self.cached_session():
        self.set_up_test_config(use_numpy, use_validation_data, with_batch_norm)
        if partial_last_batch == 'eval':
            x_train, y_train, x_eval, y_eval, x_predict = self.get_data_with_partial_last_batch_eval()
        elif partial_last_batch == 'train_and_eval':
            x_train, y_train, x_eval, y_eval, x_predict = self.get_data_with_partial_last_batch()
        else:
            x_train, y_train, x_predict = self.get_data()
            x_eval = x_train
            y_eval = y_train
        model = self.get_model(input_shapes=get_shapes(x_train))
        initial_weights = model.get_weights()
        ds_input_fn = functools.partial(self.get_input_for_correctness_test, use_numpy=use_numpy, use_validation_data=use_validation_data, with_distribution=distribution, x_train=x_train, y_train=y_train, x_eval=x_eval, y_eval=y_eval, x_predict=x_predict, training_epochs=training_epochs)
        nods_input_fn = functools.partial(self.get_input_for_correctness_test, use_numpy=use_numpy, use_validation_data=use_validation_data, with_distribution=None, x_train=x_train, y_train=y_train, x_eval=x_eval, y_eval=y_eval, x_predict=x_predict, training_epochs=training_epochs)
        results_with_ds = fit_eval_and_predict(initial_weights, input_fn=ds_input_fn, model_fn=self.get_model, distribution=distribution, is_stateful_model=is_stateful_model)
        results_without_ds = fit_eval_and_predict(initial_weights, input_fn=nods_input_fn, model_fn=self.get_model, distribution=None, is_stateful_model=is_stateful_model)
        if self.with_batch_norm == 'regular' and distribution.num_replicas_in_sync > 1:
            with self.assertRaises(AssertionError):
                compare_results(results_with_ds, results_without_ds, distribution, testcase=self, partial_last_batch=partial_last_batch)
        else:
            compare_results(results_with_ds, results_without_ds, distribution, testcase=self, partial_last_batch=partial_last_batch)