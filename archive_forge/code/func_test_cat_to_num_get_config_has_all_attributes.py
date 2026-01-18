import keras_tuner
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_cat_to_num_get_config_has_all_attributes():
    block = blocks.CategoricalToNumerical()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.CategoricalToNumerical.__init__).issubset(config.keys())