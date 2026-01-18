import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import output_adapters
def test_clf_head_transform_pd_series_to_dataset():
    adapter = output_adapters.ClassificationAdapter(name='a')
    y = adapter.adapt(pd.read_csv(test_utils.TEST_CSV_PATH).pop('survived'), batch_size=32)
    assert isinstance(y, tf.data.Dataset)