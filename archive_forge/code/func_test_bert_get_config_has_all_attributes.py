import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_bert_get_config_has_all_attributes():
    block = blocks.BertBlock()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.BertBlock.__init__).issubset(config.keys())