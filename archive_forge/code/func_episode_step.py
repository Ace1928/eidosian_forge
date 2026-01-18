import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID, TensorType
@abstractmethod
def episode_step(self, episode: Episode) -> None:
    """Increases the episode step counter (across all agents) by one.

        Args:
            episode: Episode we are stepping through.
                Useful for handling counting b/c it is called once across
                all agents that are inside this episode.
        """
    raise NotImplementedError