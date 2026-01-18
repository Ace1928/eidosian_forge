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
def run_dynamic_lr_test(self, distribution):
    with self.cached_session():
        self.set_up_test_config()
        x_train, y_train, _ = self.get_data()
        model = self.get_model(input_shapes=get_shapes(x_train))
        initial_weights = model.get_weights()
        update_freq = None
        if isinstance(distribution, tf.compat.v1.distribute.experimental.TPUStrategy) and distribution.extended.steps_per_run > 1:
            update_freq = distribution.extended.steps_per_run
        training_epochs = 2
        global_batch_size = 64
        ds_batch_size = get_batch_size(global_batch_size, distribution)
        nods_batch_size = get_batch_size(global_batch_size, None)
        ds_input_fn = functools.partial(self.get_input_for_dynamic_lr_test, x=x_train, y=y_train, batch_size=ds_batch_size, shuffle=False, epochs=training_epochs, callbacks=[LearningRateBatchScheduler(update_freq)], validation_data=(x_train, y_train))
        nods_input_fn = functools.partial(self.get_input_for_dynamic_lr_test, x=x_train, y=y_train, batch_size=nods_batch_size, shuffle=False, epochs=training_epochs, callbacks=[LearningRateBatchScheduler(update_freq)], validation_data=(x_train, y_train))
        results_with_ds = fit_eval_and_predict(initial_weights, input_fn=ds_input_fn, model_fn=self.get_model, distribution=distribution)
        results_without_ds = fit_eval_and_predict(initial_weights, input_fn=nods_input_fn, model_fn=self.get_model, distribution=None)
        compare_results(results_with_ds, results_without_ds, distribution, testcase=self)