import random
from collections import defaultdict
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.simple_list_collector import (
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvInfoDict, PolicyID, TensorType
def has_init_obs(self, agent_id: AgentID=None) -> bool:
    """Returns whether this episode has initial obs for an agent.

        If agent_id is None, return whether we have received any initial obs,
        in other words, whether this episode is completely fresh.
        """
    if agent_id is not None:
        return agent_id in self._has_init_obs and self._has_init_obs[agent_id]
    else:
        return any(list(self._has_init_obs.values()))