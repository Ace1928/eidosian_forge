import os
import shutil
import pytest
import tensorflow as tf
from tensorflow import keras
from autokeras import test_utils
from autokeras.utils import io_utils
def test_load_imdb_dataset():
    data_dir = os.path.join(os.path.dirname(keras.utils.get_file(fname='text_data', origin='https://github.com/keras-team/autokeras/releases/download/1.0.19/aclImdb_v1.tar.gz', extract=True)), 'aclImdb')
    shutil.rmtree(os.path.join(data_dir, 'train/unsup'))
    dataset = io_utils.text_dataset_from_directory(os.path.join(data_dir, 'train'), max_length=20)
    for data in dataset:
        assert data[0].dtype == tf.string
        assert data[1].dtype == tf.string
        break