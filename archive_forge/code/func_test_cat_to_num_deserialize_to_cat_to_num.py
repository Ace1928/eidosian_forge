import keras_tuner
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_cat_to_num_deserialize_to_cat_to_num():
    serialized_block = blocks.serialize(blocks.CategoricalToNumerical())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.CategoricalToNumerical)