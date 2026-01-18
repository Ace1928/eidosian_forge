import numpy as np
from typing import Optional
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
def rollouts(self, *, rollouts_per_iteration: Optional[int]=NotProvided, **kwargs) -> 'RandomAgentConfig':
    """Sets the rollout configuration.

        Args:
            rollouts_per_iteration: How many episodes to run per training iteration.

        Returns:
            This updated AlgorithmConfig object.
        """
    super().rollouts(**kwargs)
    if rollouts_per_iteration is not NotProvided:
        self.rollouts_per_iteration = rollouts_per_iteration
    return self