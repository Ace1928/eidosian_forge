from unittest import mock
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import autokeras as ak
from autokeras import keras_layers
from autokeras import test_utils
from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
@mock.patch('keras_tuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
@mock.patch('autokeras.engine.tuner.AutoTuner._prepare_model_build')
def test_final_fit_with_specified_epochs(_, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(hypermodel=test_utils.build_graph(), directory=tmp_path)
    final_fit.return_value = (mock.Mock(), mock.Mock(), mock.Mock())
    tuner.search(x=None, epochs=10, validation_data=None)
    assert final_fit.call_args_list[0][1]['epochs'] == 10