import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import output_adapters
def test_clf_head_transform_df_to_dataset():
    adapter = output_adapters.ClassificationAdapter(name='a')
    y = adapter.adapt(pd.DataFrame(test_utils.generate_one_hot_labels(dtype='np', num_classes=10)), batch_size=32)
    assert isinstance(y, tf.data.Dataset)