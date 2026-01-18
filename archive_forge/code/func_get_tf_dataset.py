import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def get_tf_dataset(self):
    from keras.src.utils.module_utils import tensorflow as tf
    output_signature = self.peek_and_get_tensor_spec()
    return tf.data.Dataset.from_generator(self.get_numpy_iterator, output_signature=output_signature)