import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from keras.src.distribute import model_combinations
def run_test_save_strategy(self, model_and_input, distribution, save_in_scope):
    """Save a model with DS."""
    saved_dir = os.path.join(self.get_temp_dir(), '3')
    with distribution.scope():
        model = model_and_input.get_model()
        x_train, y_train, _ = model_and_input.get_data()
        batch_size = model_and_input.get_batch_size()
        self._train_model(model, x_train, y_train, batch_size)
    if save_in_scope:
        with distribution.scope():
            self._save_model(model, saved_dir)
    else:
        self._save_model(model, saved_dir)
    return saved_dir