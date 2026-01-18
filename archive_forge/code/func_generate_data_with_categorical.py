import inspect
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
def generate_data_with_categorical(num_instances=100, num_numerical=10, num_categorical=3, num_classes=5, dtype='np'):
    categorical_data = np.random.randint(num_classes, size=(num_instances, num_categorical))
    numerical_data = np.random.rand(num_instances, num_numerical)
    data = np.concatenate((numerical_data, categorical_data), axis=1)
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if dtype == 'np':
        return data
    if dtype == 'dataset':
        return tf.data.Dataset.from_tensor_slices(data)