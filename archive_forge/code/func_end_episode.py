from collections import defaultdict
import logging
import time
import tree  # pip install dm_tree
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Union
import numpy as np
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv
from ray.rllib.env.external_env import ExternalEnvWrapper
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv, get_wrapper_by_cls
from ray.rllib.evaluation.collectors.simple_list_collector import _PolicyCollectorGroup
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unbatch, get_original_space
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def end_episode(self, env_id: EnvID, episode_or_exception: Union[EpisodeV2, Exception]):
    """Cleans up an episode that has finished.

        Args:
            env_id: Env ID.
            episode_or_exception: Instance of an episode if it finished successfully.
                Otherwise, the exception that was thrown,
        """
    self._callbacks.on_episode_end(worker=self._worker, base_env=self._base_env, policies=self._worker.policy_map, episode=episode_or_exception, env_index=env_id)
    for p in self._worker.policy_map.cache.values():
        if getattr(p, 'exploration', None) is not None:
            p.exploration.on_episode_end(policy=p, environment=self._base_env, episode=episode_or_exception, tf_sess=p.get_session())
    if isinstance(episode_or_exception, EpisodeV2):
        episode = episode_or_exception
        if episode.total_agent_steps == 0:
            msg = f'Data from episode {episode.episode_id} does not show any agent interactions. Hint: Make sure for at least one timestep in the episode, env.step() returns non-empty values.'
            raise ValueError(msg)
    if env_id in self._active_episodes:
        del self._active_episodes[env_id]