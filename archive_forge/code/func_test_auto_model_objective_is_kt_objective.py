from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
def test_auto_model_objective_is_kt_objective(tmp_path):
    auto_model = ak.AutoModel(ak.ImageInput(), ak.RegressionHead(), directory=tmp_path)
    assert isinstance(auto_model.objective, keras_tuner.Objective)