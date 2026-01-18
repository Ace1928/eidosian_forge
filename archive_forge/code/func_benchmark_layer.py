import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing.benchmarks import (
from tensorflow.python.eager.def_function import (
def benchmark_layer(self):
    for batch in BATCH_SIZES:
        name = f'embedding|varlen|batch_{batch}'
        k_time, f_time = embedding_varlen(batch_size=batch, max_length=256)
        self.report(name, k_time, f_time, NUM_REPEATS)