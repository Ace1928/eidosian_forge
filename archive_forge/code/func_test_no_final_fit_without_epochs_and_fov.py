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
@mock.patch('autokeras.engine.tuner.AutoTuner.get_best_models', return_value=[mock.Mock()])
@mock.patch('autokeras.engine.tuner.AutoTuner._prepare_model_build')
@mock.patch('autokeras.pipeline.load_pipeline')
@mock.patch('keras_tuner.Oracle.get_best_trials', return_value=[mock.Mock()])
def test_no_final_fit_without_epochs_and_fov(_, _1, _2, get_best_models, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(hypermodel=test_utils.build_graph(), directory=tmp_path)
    tuner.search(x=None, epochs=None, validation_data=None)
    final_fit.assert_not_called()