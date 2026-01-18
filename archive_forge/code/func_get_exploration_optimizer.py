from gymnasium.spaces import Space
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.framework import try_import_torch, TensorType
from ray.rllib.utils.typing import LocalOptimizer, AlgorithmConfigDict
@DeveloperAPI
def get_exploration_optimizer(self, optimizers: List[LocalOptimizer]) -> List[LocalOptimizer]:
    """May add optimizer(s) to the Policy's own `optimizers`.

        The number of optimizers (Policy's plus Exploration's optimizers) must
        match the number of loss terms produced by the Policy's loss function
        and the Exploration component's loss terms.

        Args:
            optimizers: The list of the Policy's local optimizers.

        Returns:
            The updated list of local optimizers to use on the different
            loss terms.
        """
    return optimizers