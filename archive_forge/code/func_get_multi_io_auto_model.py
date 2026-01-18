from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
def get_multi_io_auto_model(tmp_path):
    return ak.AutoModel([ak.ImageInput(), ak.ImageInput()], [ak.RegressionHead(), ak.RegressionHead()], directory=tmp_path, max_trials=2, overwrite=False)