import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_conv_deserialize_to_conv():
    serialized_block = blocks.serialize(blocks.ConvBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.ConvBlock)