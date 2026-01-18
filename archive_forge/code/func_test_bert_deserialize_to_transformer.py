import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_bert_deserialize_to_transformer():
    serialized_block = blocks.serialize(blocks.BertBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.BertBlock)