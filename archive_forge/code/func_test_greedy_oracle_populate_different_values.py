from unittest import mock
import keras_tuner
from tensorflow import keras
import autokeras as ak
from autokeras import test_utils
from autokeras.tuners import greedy
from autokeras.tuners import task_specific
@mock.patch('autokeras.tuners.greedy.GreedyOracle.get_best_trials')
def test_greedy_oracle_populate_different_values(get_best_trials):
    hp = keras_tuner.HyperParameters()
    test_utils.build_graph().build(hp)
    oracle = greedy.GreedyOracle(objective='val_loss', seed=test_utils.SEED)
    trial = mock.Mock()
    trial.hyperparameters = hp
    get_best_trials.return_value = [trial]
    oracle.update_space(hp)
    values_a = oracle.populate_space('a')['values']
    values_b = oracle.populate_space('b')['values']
    assert not all([values_a[key] == values_b[key] for key in values_a])