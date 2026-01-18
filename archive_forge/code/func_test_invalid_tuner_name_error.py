from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
def test_invalid_tuner_name_error(tmp_path):
    with pytest.raises(ValueError) as info:
        ak.AutoModel(ak.ImageInput(), ak.RegressionHead(), directory=tmp_path, tuner='unknown')
    assert 'Expected the tuner argument to be one of' in str(info.value)