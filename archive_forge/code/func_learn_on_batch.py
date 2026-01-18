import gymnasium as gym
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AlgorithmConfigDict
@override(RandomPolicy)
def learn_on_batch(self, samples):
    if self._leakage_size == 'small':
        self._leak.append(False)
    else:
        self._leak.append([False] * 100)
    return super().learn_on_batch(samples)