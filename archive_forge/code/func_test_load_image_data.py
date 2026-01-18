import os
import shutil
import pytest
import tensorflow as tf
from tensorflow import keras
from autokeras import test_utils
from autokeras.utils import io_utils
def test_load_image_data():
    dataset = io_utils.image_dataset_from_directory(IMG_DATA_DIR, image_size=(180, 180), validation_split=0.2, subset='training', seed=test_utils.SEED)
    val_dataset = io_utils.image_dataset_from_directory(IMG_DATA_DIR, image_size=(180, 180), validation_split=0.2, subset='validation', seed=test_utils.SEED)
    for data in dataset:
        assert data[0].numpy().shape == (32, 180, 180, 3)
        assert data[1].dtype == tf.string
        break
    for data in val_dataset:
        assert data[0].numpy().shape == (32, 180, 180, 3)
        assert data[1].dtype == tf.string
        break