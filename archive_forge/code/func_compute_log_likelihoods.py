import gymnasium as gym
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AlgorithmConfigDict
@override(RandomPolicy)
def compute_log_likelihoods(self, *args, **kwargs):
    if self._leakage_size == 'small':
        self._leak.append('test')
    else:
        self._leak.append(['test'] * 100)
    return super().compute_log_likelihoods(*args, **kwargs)