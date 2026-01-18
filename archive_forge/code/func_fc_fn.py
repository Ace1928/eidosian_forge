import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing.benchmarks import (
from tensorflow.python.eager.def_function import (
@tf_function()
def fc_fn(tensors):
    fc.transform_feature(tf.__internal__.feature_column.FeatureTransformationCache(tensors), None)