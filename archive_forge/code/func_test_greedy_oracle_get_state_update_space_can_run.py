from unittest import mock
import keras_tuner
from tensorflow import keras
import autokeras as ak
from autokeras import test_utils
from autokeras.tuners import greedy
from autokeras.tuners import task_specific
def test_greedy_oracle_get_state_update_space_can_run():
    oracle = greedy.GreedyOracle(objective='val_loss')
    oracle.set_state(oracle.get_state())
    hp = keras_tuner.HyperParameters()
    hp.Boolean('test')
    oracle.update_space(hp)