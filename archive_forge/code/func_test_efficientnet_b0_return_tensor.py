import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_efficientnet_b0_return_tensor():
    block = blocks.EfficientNetBlock(version='b0', pretrained=False)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1