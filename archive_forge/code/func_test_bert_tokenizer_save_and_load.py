import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_bert_tokenizer_save_and_load(tmp_path):
    x_train, x_test, y_train = get_text_data()
    max_sequence_length = 8
    layer = layer_module.BertTokenizer(max_sequence_length=max_sequence_length)
    input_node = keras.Input(shape=(1,), dtype=tf.string)
    output_node = layer(input_node)
    model = keras.Model(input_node, output_node)
    model.save(os.path.join(tmp_path, 'model'))
    model2 = keras.models.load_model(os.path.join(tmp_path, 'model'))
    assert np.array_equal(model.predict(x_train), model2.predict(x_train))