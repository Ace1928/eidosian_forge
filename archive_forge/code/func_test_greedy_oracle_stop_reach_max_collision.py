from unittest import mock
import keras_tuner
from tensorflow import keras
import autokeras as ak
from autokeras import test_utils
from autokeras.tuners import greedy
from autokeras.tuners import task_specific
@mock.patch('autokeras.tuners.greedy.GreedyOracle._compute_values_hash')
@mock.patch('autokeras.tuners.greedy.GreedyOracle.get_best_trials')
def test_greedy_oracle_stop_reach_max_collision(get_best_trials, compute_values_hash):
    hp = keras_tuner.HyperParameters()
    test_utils.build_graph().build(hp)
    oracle = greedy.GreedyOracle(objective='val_loss', seed=test_utils.SEED)
    trial = mock.Mock()
    trial.hyperparameters = hp
    get_best_trials.return_value = [trial]
    compute_values_hash.return_value = 1
    oracle.update_space(hp)
    oracle.populate_space('a')['values']
    assert oracle.populate_space('b')['status'] == keras_tuner.engine.trial.TrialStatus.STOPPED