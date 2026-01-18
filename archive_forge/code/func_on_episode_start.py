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
def on_episode_start(self, policy: 'Policy', *, environment: BaseEnv=None, episode: int=None, tf_sess: Optional['tf.Session']=None):
    """Handles necessary exploration logic at the beginning of an episode.

        Args:
            policy: The Policy object that holds this Exploration.
            environment: The environment object we are acting in.
            episode: The number of the episode that is starting.
            tf_sess: In case of tf, the session object.
        """
    pass