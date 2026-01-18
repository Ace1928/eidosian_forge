from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
def test_auto_model_project_name_field_as_specified(tmp_path):
    auto_model = ak.AutoModel(ak.ImageInput(), ak.RegressionHead(), directory=tmp_path, project_name='auto_model')
    assert auto_model.project_name == 'auto_model'