import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from keras.src.distribute import model_combinations
def load_and_run_with_saved_model_api(distribution, saved_dir, predict_dataset, output_name):
    """Loads a saved_model using tf.saved_model API, and runs it."""
    func = tf.saved_model.load(saved_dir)
    if distribution:
        dist_predict_dataset = distribution.experimental_distribute_dataset(predict_dataset)
        per_replica_predict_data = next(iter(dist_predict_dataset))
        result = distribution.run(func.signatures[_DEFAULT_FUNCTION_KEY], args=(per_replica_predict_data,))
        result = result[output_name]
        reduced = distribution.experimental_local_results(result)
        concat = tf.concat(reduced, 0)
        return concat
    else:
        result = func.signatures[_DEFAULT_FUNCTION_KEY](next(iter(predict_dataset)))
        return result[output_name]