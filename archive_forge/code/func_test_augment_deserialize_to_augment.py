import keras_tuner
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_augment_deserialize_to_augment():
    serialized_block = blocks.serialize(blocks.ImageAugmentation(zoom_factor=0.1, contrast_factor=hyperparameters.Float('contrast_factor', 0.1, 0.5)))
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.ImageAugmentation)
    assert block.zoom_factor == 0.1
    assert isinstance(block.contrast_factor, hyperparameters.Float)