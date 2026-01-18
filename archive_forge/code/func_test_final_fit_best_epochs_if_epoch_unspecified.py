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
@mock.patch('autokeras.engine.tuner.AutoTuner._get_best_trial_epochs', return_value=2)
@mock.patch('autokeras.engine.tuner.AutoTuner._prepare_model_build')
def test_final_fit_best_epochs_if_epoch_unspecified(_, best_epochs, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(hypermodel=test_utils.build_graph(), directory=tmp_path)
    final_fit.return_value = (mock.Mock(), mock.Mock(), mock.Mock())
    tuner.search(x=mock.Mock(), epochs=None, validation_split=0.2, validation_data=mock.Mock())
    assert final_fit.call_args_list[0][1]['epochs'] == 2