import gc
import os
import platform
import tracemalloc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.exploration.random_encoder import (
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.tune.callback import _CallbackMeta
import psutil
@override(DefaultCallbacks)
def on_episode_created(self, *, worker: 'RolloutWorker', base_env: BaseEnv, policies: Dict[PolicyID, Policy], env_index: int, episode: Union[Episode, EpisodeV2], **kwargs) -> None:
    for callback in self._callback_list:
        callback.on_episode_created(worker=worker, base_env=base_env, policies=policies, env_index=env_index, episode=episode, **kwargs)