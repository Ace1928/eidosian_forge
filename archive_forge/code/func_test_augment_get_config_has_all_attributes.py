import keras_tuner
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_augment_get_config_has_all_attributes():
    block = blocks.ImageAugmentation()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.ImageAugmentation.__init__).issubset(config.keys())