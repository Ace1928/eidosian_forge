import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_merge_deserialize_to_merge():
    serialized_block = blocks.serialize(blocks.Merge())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.Merge)