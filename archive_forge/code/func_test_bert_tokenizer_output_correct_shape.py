import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_bert_tokenizer_output_correct_shape(tmp_path):
    x_train, x_test, y_train = get_text_data()
    max_sequence_length = 8
    token_layer = layer_module.BertTokenizer(max_sequence_length=max_sequence_length)
    output = token_layer(x_train)
    assert output[0].shape == (x_train.shape[0], max_sequence_length)
    assert output[1].shape == (x_train.shape[0], max_sequence_length)
    assert output[2].shape == (x_train.shape[0], max_sequence_length)