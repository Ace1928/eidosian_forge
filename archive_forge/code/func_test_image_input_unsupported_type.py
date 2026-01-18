import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import input_adapters
from autokeras.utils import data_utils
def test_image_input_unsupported_type():
    x = 'unknown'
    adapter = input_adapters.ImageAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert 'Expect the data to ImageInput to be numpy' in str(info.value)