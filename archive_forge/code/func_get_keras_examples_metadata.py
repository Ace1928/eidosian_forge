import timeit
import numpy as np
from keras.src import callbacks
from keras.src.benchmarks import distribution_util
def get_keras_examples_metadata(keras_model, batch_size, impl='.keras.cfit_graph'):
    return {'model_name': 'keras_examples', 'implementation': keras_model + impl, 'parameters': 'bs_' + str(batch_size)}