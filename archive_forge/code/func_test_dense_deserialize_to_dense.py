import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_dense_deserialize_to_dense():
    serialized_block = blocks.serialize(blocks.DenseBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.DenseBlock)