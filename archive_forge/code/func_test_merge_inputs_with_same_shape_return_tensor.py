import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_merge_inputs_with_same_shape_return_tensor():
    block = blocks.Merge()
    outputs = block.build(keras_tuner.HyperParameters(), [keras.Input(shape=(32,), dtype=tf.float32), keras.Input(shape=(32,), dtype=tf.float32)])
    assert len(nest.flatten(outputs)) == 1