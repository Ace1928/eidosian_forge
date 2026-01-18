import os
import shutil
import pytest
import tensorflow as tf
from tensorflow import keras
from autokeras import test_utils
from autokeras.utils import io_utils
def test_load_image_data_raise_subset_error():
    with pytest.raises(ValueError) as info:
        io_utils.image_dataset_from_directory(IMG_DATA_DIR, image_size=(180, 180), validation_split=0.2, subset='abcd', seed=test_utils.SEED)
    assert '`subset` must be either' in str(info.value)